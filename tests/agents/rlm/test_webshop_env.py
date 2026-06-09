"""Tests for :class:`WebShopEnv` (full-scope agentic envs, spec §5, 2026-06-01).

The contract under test (all on the base venv, NO network, NO ``web_agent_site``):

* ``available()`` reports ``(False, reason)`` when ``WEBSHOP_URL`` is unset — and
  never raises.
* A full episode against an INJECTED FAKE server: reset → ``search[red shoes]``
  lists items → ``click[<item>]`` opens it → ``click[buy now]`` ends the episode
  with the fake's matching-score reward, ``done=True`` and a populated ``info``.
* A malformed action returns a nudge observation and does NOT raise / does NOT end
  the episode (it wastes a turn — the fail-soft contract of ``AgenticEnv``).
* If the env is used while the server is unreachable, the first ``step`` finishes
  with ``info={"unavailable": True, ...}`` — never a raise.

The fake is a tiny in-memory WebShop that speaks the same JSON ``observation /
reward / done`` shape the gym-style server step endpoint returns, so the env's
transport seam is exercised end-to-end without touching the real ``urllib`` path.
"""

from __future__ import annotations

import json
import sys

import pytest

# Mirror the agent's FLAT-import sandbox: the copyable env modules resolve each
# other as top-level modules sitting next to sdar_env_base.py. Remove the flat dir
# right after import so it does not leak into the rest of the session (a lingering
# entry gives package modules like rubric_guard a second identity, breaking
# unrelated tests such as rl_scaffold's RubricGuardFailure assertion).
_RLM_DIR = str(  # repo-relative: the old hardcoded /home/sww35/... path
    __import__("pathlib").Path(__file__).resolve().parents[3] / "backend" / "agents" / "rlm"
)  # only collected on the author's machine (audit 2026-06-09)
sys.path.insert(0, _RLM_DIR)
try:
    import webshop_env  # noqa: E402  (after the sys.path insert, by design)
    from webshop_env import HttpResponse, WebShopEnv  # noqa: E402
finally:
    while _RLM_DIR in sys.path:
        sys.path.remove(_RLM_DIR)

# The copyable module must resolve as a top-level module (the agent's FLAT import).
assert webshop_env.WebShopEnv is WebShopEnv
assert webshop_env.WebShopEnv.max_turns == 15


# --------------------------------------------------------------------------- #
# A fake WebShop server: an in-memory catalog reachable through the same        #
# get(url)/post(url, data) seam the real UrllibClient implements.               #
# --------------------------------------------------------------------------- #
class FakeWebShopClient:
    """In-memory WebShop returning the JSON page shape the env coerces.

    Routes mirrored from ``web_agent_site``:
      GET  /<session>?goal_idx=N           -> instruction + search box (landing)
      POST /search_results/<session>       -> a numbered results list
      POST /click/<session> {button: id}   -> item page (with options + Buy Now)
      POST /click/<session> {done: True}   -> purchase -> reward
    """

    BUY_REWARD = 0.875  # the matching score the fake awards on a correct buy

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []  # (verb, url, data)
        self._purchased = False

    # -- helpers --
    @staticmethod
    def _json(payload: dict) -> HttpResponse:
        return HttpResponse(text=json.dumps(payload), status=200)

    @staticmethod
    def _path(url: str) -> str:
        # Strip scheme://host and any query string -> just the path.
        no_q = url.split("?", 1)[0]
        if "://" in no_q:
            no_q = "/" + no_q.split("://", 1)[1].split("/", 1)[1] if "/" in no_q.split("://", 1)[1] else "/"
        return no_q

    # -- transport seam --
    def get(self, url: str) -> HttpResponse:
        self.calls.append(("GET", url, None))
        path = self._path(url)
        if path.startswith("/done"):
            return self._json({"observation": "Order placed.", "reward": self.BUY_REWARD, "done": True})
        # Landing page for the session.
        return self._json(
            {
                "observation": (
                    "Instruction: i want a pair of red running shoes, and price "
                    "lower than 50 dollars\n[Search]"
                ),
                "reward": 0.0,
                "done": False,
            }
        )

    def post(self, url: str, data: dict | None = None) -> HttpResponse:
        self.calls.append(("POST", url, data))
        path = self._path(url)
        data = data or {}

        if path.startswith("/search_results"):
            q = data.get("search_query") or data.get("keywords") or ""
            return self._json(
                {
                    "observation": (
                        f"Search results for '{q}':\n"
                        "[B001] Red Running Shoes - $42.00\n"
                        "[B002] Blue Sneakers - $55.00\n"
                        "[next >]"
                    ),
                    "reward": 0.0,
                    "done": False,
                }
            )

        if path.startswith("/click"):
            button = (data.get("button") or data.get("click") or "").strip().lower()
            is_done = bool(data.get("done")) or button in {"buy now", "buy"}
            if is_done:
                self._purchased = True
                return self._json(
                    {
                        "observation": "Thank you for your purchase!",
                        "reward": self.BUY_REWARD,
                        "done": True,
                    }
                )
            # Non-terminal click -> an item page with options + the Buy Now button.
            return self._json(
                {
                    "observation": (
                        "Red Running Shoes - $42.00\n"
                        "Size: [8] [9] [10]\nColor: [red] [black]\n"
                        "[description] [features] [Buy Now]"
                    ),
                    "reward": 0.0,
                    "done": False,
                }
            )

        return HttpResponse(text="", status=404)


@pytest.fixture(autouse=True)
def _clear_webshop_url(monkeypatch):
    """Every test starts with WEBSHOP_URL unset; tests opt back in via base_url."""
    monkeypatch.delenv("WEBSHOP_URL", raising=False)


# --------------------------------------------------------------------------- #
# available()                                                                   #
# --------------------------------------------------------------------------- #
def test_available_false_when_url_unset():
    ok, reason = WebShopEnv.available()
    assert ok is False
    assert "WEBSHOP_URL" in reason


def test_available_does_not_raise_on_dead_probe(monkeypatch):
    monkeypatch.setenv("WEBSHOP_URL", "http://127.0.0.1:3000")

    class DeadClient:
        def get(self, url):
            return HttpResponse(text="", status=0)

        def post(self, url, data=None):  # pragma: no cover - unused
            return HttpResponse(text="", status=0)

    ok, reason = WebShopEnv.available(client=DeadClient())
    assert ok is False
    assert "status 0" in reason


def test_available_true_when_probe_succeeds(monkeypatch):
    monkeypatch.setenv("WEBSHOP_URL", "http://127.0.0.1:3000")
    ok, reason = WebShopEnv.available(client=FakeWebShopClient())
    assert ok is True
    assert "reachable" in reason


# --------------------------------------------------------------------------- #
# Full episode against the fake server                                          #
# --------------------------------------------------------------------------- #
def test_full_episode_search_click_buy_yields_reward():
    fake = FakeWebShopClient()
    env = WebShopEnv(base_url="http://fake-webshop", client=fake)

    # reset -> instruction + search box
    obs0 = env.reset(seed=1, task=0)
    assert "Instruction" in obs0
    assert env.done is False
    assert env.turns_taken == 0

    # search[red shoes] -> a numbered item list
    r1 = env.step("search[red shoes]")
    assert r1.done is False
    assert r1.reward == 0.0
    assert "B001" in r1.observation and "Red Running Shoes" in r1.observation
    assert env.turns_taken == 1

    # click[B001] -> item page with options + Buy Now
    r2 = env.step("click[B001]")
    assert r2.done is False
    assert "Buy Now" in r2.observation
    assert env.turns_taken == 2

    # click[buy now] -> terminal, matching-score reward from the fake
    r3 = env.step("click[buy now]")
    assert r3.done is True
    assert r3.reward == pytest.approx(FakeWebShopClient.BUY_REWARD)
    assert r3.info["success"] is True  # 0.875 > 0.5
    assert r3.info["reward"] == pytest.approx(FakeWebShopClient.BUY_REWARD)
    assert r3.info["steps"] == 3
    assert env.done is True
    assert env.episode_reward() == pytest.approx(FakeWebShopClient.BUY_REWARD)
    assert env.last_info["success"] is True

    # The transcript the trainer renders carries the whole interaction.
    transcript = env.build_student_prompt()
    assert "search[red shoes]" in transcript
    assert "click[buy now]" in transcript


def test_buy_falls_back_to_done_route_when_click_omits_reward():
    """If the buy click page carries no reward, the env reads the /done route."""

    class NoRewardOnBuyClient(FakeWebShopClient):
        def post(self, url, data=None):
            self.calls.append(("POST", url, data))
            path = self._path(url)
            data = data or {}
            if path.startswith("/click") and (data.get("done") or "buy" in (data.get("button") or "").lower()):
                # Terminal page but NO reward field -> env must hit /done.
                return self._json({"observation": "Thanks!", "done": True})
            return super().post(url, data)

    fake = NoRewardOnBuyClient()
    env = WebShopEnv(base_url="http://fake-webshop", client=fake)
    env.reset(seed=0, task=0)
    res = env.step("click[buy now]")
    assert res.done is True
    assert res.reward == pytest.approx(FakeWebShopClient.BUY_REWARD)
    # Confirm the env actually queried the /done fallback route.
    assert any(v == "GET" and "/done" in u for v, u, _ in fake.calls)


# --------------------------------------------------------------------------- #
# Fail-soft behaviours                                                          #
# --------------------------------------------------------------------------- #
def test_malformed_action_nudges_without_raising():
    fake = FakeWebShopClient()
    env = WebShopEnv(base_url="http://fake-webshop", client=fake)
    env.reset(seed=0, task=0)

    res = env.step("uhh I think I should look at shoes maybe???")  # no valid verb
    assert res.done is False
    assert res.reward == 0.0
    assert "search[" in res.observation and "click[" in res.observation  # the nudge
    # It still consumed a turn (defensive parsing wastes the turn, never crashes).
    assert env.turns_taken == 1


def test_action_parsing_tolerates_fences_colons_and_prose():
    fake = FakeWebShopClient()
    env = WebShopEnv(base_url="http://fake-webshop", client=fake)
    env.reset(seed=0, task=0)

    # Code-fenced + narrated search still parses to a real search.
    res = env.step("Let me search.\n```\nsearch[red shoes]\n```")
    assert res.done is False
    assert "Search results" in res.observation

    # Colon syntax parses too.
    verb, arg = WebShopEnv._parse_action("search: blue sneakers")
    assert verb == "search" and arg == "blue sneakers"

    # Bare buy synonyms map to the terminal action.
    assert WebShopEnv._is_buy("Buy Now") is True
    assert WebShopEnv._is_buy("purchase") is True
    assert WebShopEnv._is_buy("description") is False


def test_unavailable_server_first_step_is_terminal_not_raise():
    # No base_url and WEBSHOP_URL unset -> the env is unavailable.
    env = WebShopEnv(base_url="", client=FakeWebShopClient())
    obs = env.reset(seed=0, task=0)
    assert "unavailable" in obs.lower()

    res = env.step("search[anything]")  # must not raise
    assert res.done is True
    assert res.reward == 0.0
    assert res.info.get("unavailable") is True
    assert "reason" in res.info


def test_set_but_dead_url_first_step_is_terminal_unavailable():
    """WEBSHOP_URL is set but the server is DOWN (status 0 on every call).

    The Codex-HIGH fix: a set-but-dead URL must return the SAME terminal
    unavailable result on the FIRST step that the empty-URL path returns — it must
    NOT burn its whole turn budget on status-0 degraded pages.
    """

    class DeadClient:
        def __init__(self):
            self.calls = 0

        def get(self, url):
            self.calls += 1
            return HttpResponse(text="", status=0)  # server unreachable

        def post(self, url, data=None):
            self.calls += 1
            return HttpResponse(text="", status=0)

    dead = DeadClient()
    # A non-empty base_url -> the empty-URL guard does NOT fire; the unavailability
    # must be detected via reset()'s failed landing instead.
    env = WebShopEnv(base_url="http://127.0.0.1:3000", client=dead)
    obs = env.reset(seed=0, task=0)
    assert "WEBSHOP_URL" not in obs  # not the empty-URL message; it's the degraded landing
    calls_after_reset = dead.calls

    res = env.step("search[red shoes]")  # FIRST step -> terminal unavailable
    assert res.done is True
    assert res.reward == 0.0
    assert res.info.get("unavailable") is True
    assert "reason" in res.info
    # It returned the terminal WITHOUT issuing a fresh search POST (no burned turn).
    assert dead.calls == calls_after_reset


def test_set_but_dead_url_does_not_burn_all_turns():
    """A second step against a dead-after-reset server is STILL terminal/no-op.

    Regression for the burn-all-turns bug: even if the policy keeps emitting
    actions, every step short-circuits to the unavailable terminal instead of
    POSTing degraded pages turn after turn.
    """

    class DeadClient:
        def __init__(self):
            self.posts = 0

        def get(self, url):
            return HttpResponse(text="", status=0)

        def post(self, url, data=None):
            self.posts += 1
            return HttpResponse(text="", status=0)

    dead = DeadClient()
    env = WebShopEnv(base_url="http://127.0.0.1:3000", client=dead)
    env.reset(seed=0, task=0)

    for _ in range(5):
        res = env.step("search[shoes]")
        assert res.done is True
        assert res.info.get("unavailable") is True
    # No action ever POSTed to the dead server (it short-circuited every time).
    assert dead.posts == 0


def test_client_that_raises_after_reset_is_terminal_unavailable():
    """A client that *raises* (not just status 0) is still fail-soft + terminal.

    The injectable seam (`_get`/`_post`) swallows the exception into status 0, so a
    raising client lands in the same unavailable terminal — no exception escapes.
    """

    class RaisingClient:
        def get(self, url):
            raise ConnectionError("connection refused")

        def post(self, url, data=None):  # pragma: no cover - never reached past reset
            raise ConnectionError("connection refused")

    env = WebShopEnv(base_url="http://127.0.0.1:3000", client=RaisingClient())
    obs = env.reset(seed=0, task=0)  # must not raise
    assert isinstance(obs, str)

    res = env.step("search[anything]")  # must not raise
    assert res.done is True
    assert res.reward == 0.0
    assert res.info.get("unavailable") is True


def test_reset_with_garbage_seed_does_not_raise():
    """The Codex-MEDIUM fix: a non-int / garbage seed must never raise.

    Both reset() (session-id derivation) and _resolve_goal_idx() (seed fallback)
    coerce the seed defensively; a string / float / object seed degrades to a
    deterministic fallback instead of propagating a TypeError/ValueError.
    """
    fake = FakeWebShopClient()
    env = WebShopEnv(base_url="http://fake-webshop", client=fake)

    for bad_seed in ("not-a-number", 3.7, object(), [], {"x": 1}):
        obs = env.reset(seed=bad_seed, task=None)  # must not raise
        assert isinstance(obs, str)
        assert env.done is False
        assert env.turns_taken == 0
        # The session id is still well-formed (no exception leaked into it).
        assert env._session_id.startswith("sdar_")

    # A numeric string seed still parses to its int value (coercion, not discard).
    env.reset(seed="42", task=None)
    assert env._session_id == "sdar_42_42"


def test_resolve_goal_idx_with_garbage_seed_does_not_raise():
    """_resolve_goal_idx() falls back safely when seed is non-int and task is None."""
    env = WebShopEnv(base_url="http://x", client=FakeWebShopClient(), num_goals=10)
    # task=None forces the seed fallback path; a garbage seed -> deterministic 0.
    assert env._resolve_goal_idx(task=None, seed="garbage") == 0
    assert env._resolve_goal_idx(task=None, seed=3.9) == 3      # int(3.9) == 3
    assert env._resolve_goal_idx(task=None, seed=object()) == 0


def test_degraded_response_does_not_crash_episode():
    """A non-2xx server response degrades to an observation, not an exception."""

    class FlakyClient:
        def get(self, url):
            return HttpResponse(text="<html>ok</html>", status=200)

        def post(self, url, data=None):
            return HttpResponse(text="", status=500)  # server error on every action

    env = WebShopEnv(base_url="http://flaky", client=FlakyClient())
    env.reset(seed=0, task=0)
    res = env.step("search[shoes]")
    assert res.done is False  # degraded, but the episode continues
    assert "degraded" in res.observation.lower()
    assert env.turns_taken == 1


def test_turn_budget_exhaustion_ends_episode_zero_reward():
    """Reaching max_turns without buying ends the episode with reward 0.0."""
    fake = FakeWebShopClient()
    env = WebShopEnv(base_url="http://fake-webshop", client=fake)
    env.max_turns = 3  # shrink for the test
    env.reset(seed=0, task=0)

    env.step("search[shoes]")          # turn 1
    env.step("click[B001]")            # turn 2
    res = env.step("click[description]")  # turn 3 -> budget exhausted
    assert res.done is True
    assert res.reward == 0.0
    assert env.episode_reward() == 0.0


# --------------------------------------------------------------------------- #
# HTML coercion + injected get/post-callable seam                               #
# --------------------------------------------------------------------------- #
def test_http_get_post_callable_seam_and_html_landing():
    """The two-callable seam works and an HTML landing page is stripped to text."""
    seen = {"gets": [], "posts": []}

    def http_get(url):
        seen["gets"].append(url)
        return HttpResponse(
            text="<html><body>Instruction: buy a <b>red</b> mug<button>Search</button></body></html>",
            status=200,
        )

    def http_post(url, data):
        seen["posts"].append((url, data))
        return HttpResponse(
            text=json.dumps({"observation": "[B009] Red Mug - $9", "reward": 0.0, "done": False}),
            status=200,
        )

    env = WebShopEnv(base_url="http://html-shop", http_get=http_get, http_post=http_post)
    obs0 = env.reset(seed=2, task={"goal_idx": 7})
    assert "Instruction" in obs0 and "red" in obs0  # HTML tags stripped, text kept
    assert "<" not in obs0  # tags gone

    res = env.step("search[red mug]")
    assert "B009" in res.observation
    assert seen["gets"] and seen["posts"]


def test_task_index_resolution_is_deterministic():
    env = WebShopEnv(base_url="http://x", client=FakeWebShopClient(), num_goals=10)
    assert env._resolve_goal_idx(task=3, seed=None) == 3
    assert env._resolve_goal_idx(task={"index": 25}, seed=None) == 5  # 25 % 10
    assert env._resolve_goal_idx(task="-12", seed=None) == 2          # abs, then % 10
    assert env._resolve_goal_idx(task=None, seed=7) == 7              # seed fallback
    assert env._resolve_goal_idx(task=None, seed=None) == 0           # default
