"""WebShopEnv — a real multi-turn agentic shopping environment for SDAR.

The 2026-05-31 SDAR run (`prj_09047604e591d969`) de-scoped WebShop entirely: it is
the canonical web-navigation benchmark of the paper's three environments and, like
ALFWorld, cannot be faked as closed-book QA — the policy has to *search* a catalog,
*click* through item lists and option pages, and *buy* the right product to earn the
WebShop matching reward.  This module restores it as a real :class:`AgenticEnv`
talking to the WebShop server that ``env_cache.acquire_webshop`` stands up and
exports as ``WEBSHOP_URL``.

**Action grammar (WebShop canonical).**  The policy emits ONE action per turn:

* ``search[<query>]`` — issue a keyword search; the observation becomes the search
  results page (a numbered list of items with their clickable buttons).
* ``click[<target>]`` — click a button on the current page.  ``<target>`` is the
  item id / ASIN, an option value (``click[red]``), a navigation button
  (``click[< prev]``, ``click[next >]``), ``click[description]`` / ``click[features]``,
  or the terminal ``click[buy now]``.  Clicking *buy now* ends the episode and the
  server returns the matching reward.

Parsing is deliberately forgiving (case-insensitive verb, tolerant of code fences,
stray prose, and ``search: q`` / ``click: x`` colon syntax) because a real policy's
output is noisy — a malformed action wastes one turn and returns a nudge
observation, it never raises (the fail-soft contract of §0 / the ``AgenticEnv``
docstring).

**Reward.**  The terminal reward is the server's WebShop *matching score* — a float
in ``[0, 1]`` measuring how well the purchased item + chosen options match the
instruction's goal (attributes, price cap, options).  ``info`` on the terminal step
is ``{"success": reward > 0.5, "reward": reward, "steps": turns_taken}``.

**HTTP, defensively.**  The real transport is stdlib ``urllib.request`` (no extra
dependency — more robust than ``requests`` inside a bare sandbox), every call wrapped
with a timeout + try/except that degrades to a textual observation rather than
propagating.  The transport is an injectable seam (``http_get`` / ``http_post``
callables, or a ``client`` object exposing ``.get`` / ``.post``) so the unit test
drives a fully in-memory FAKE server with zero network.

**Availability.**  :meth:`available` reports whether ``WEBSHOP_URL`` is set and a
quick GET succeeds.  If the env is constructed and used anyway while the server is
unreachable, the first :meth:`step` returns a terminal zero-reward step carrying
``info={"unavailable": True, "reason": ...}`` — never a raise.  (The harness already
converts an unavailable env into a verified rubric Exclusion in ``env_cache``; this
is the in-cell safety net for the case where the server dies after provisioning.)

Copyable helper — mirror of the ``gpu_cell_runner.py`` / ``sdar_env_base.py`` /
``rubric_guard.py`` pattern.  ``run_with_sdk`` copies this file into
``code/webshop_env.py`` and the agent imports it with the FLAT import
``from sdar_env_base import AgenticEnv, StepResult``.  Heavy/optional deps stay out
of import scope: there are none here beyond the Python stdlib.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

# FLAT import: this file is copied next to sdar_env_base.py inside code/, so the
# agent-generated trainer and this module both resolve it as a top-level module.
from sdar_env_base import AgenticEnv, StepResult

__all__ = ["WebShopEnv", "HttpResponse", "UrllibClient"]

#: Default per-request HTTP timeout (seconds).  Kept short so a wedged server
#: degrades into a textual observation quickly instead of stalling a cell.
_DEFAULT_TIMEOUT_S = 10.0

#: Cap the page text fed into the prompt so a giant catalog page cannot blow the
#: token budget; the head of a WebShop page carries the instruction + the actionable
#: buttons, which is what the policy needs.
_MAX_OBS_CHARS = 4000

# Action verbs the policy may emit.  WebShop's canonical grammar is exactly two.
_SEARCH = "search"
_CLICK = "click"


@dataclass
class HttpResponse:
    """A minimal HTTP response the env understands.

    ``text`` is the page body (HTML or, for the JSON-ish dev server, a JSON string).
    ``status`` is the HTTP status code; ``ok`` is ``200 <= status < 400``.  Both the
    real ``UrllibClient`` and the test's FAKE client return this shape, so the env
    never branches on the transport.
    """

    text: str = ""
    status: int = 0

    @property
    def ok(self) -> bool:
        return 200 <= int(self.status) < 400


class HttpClient(Protocol):
    """Transport seam: anything with ``get(url)`` / ``post(url, data)`` works.

    The default :class:`UrllibClient` hits a real server; the unit test injects a
    fake implementing the same two methods over an in-memory catalog.
    """

    def get(self, url: str) -> HttpResponse: ...  # noqa: E704
    def post(self, url: str, data: dict[str, Any] | None = None) -> HttpResponse: ...  # noqa: E704


class UrllibClient:
    """Stdlib ``urllib.request`` HTTP client — the real transport.

    Every call is wrapped: a timeout, then a broad ``except`` that returns an
    ``HttpResponse(status=0)`` (``ok == False``).  The env turns a non-ok response
    into a degraded observation, so a transient server hiccup costs one noisy turn
    rather than crashing the cell.
    """

    def __init__(self, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = float(timeout_s)

    def _request(self, url: str, data: bytes | None = None) -> HttpResponse:
        try:
            req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                body = resp.read()
                status = int(getattr(resp, "status", 0) or getattr(resp, "code", 0) or 0)
                text = body.decode("utf-8", errors="replace") if isinstance(body, (bytes, bytearray)) else str(body)
                return HttpResponse(text=text, status=status or 200)
        except urllib.error.HTTPError as exc:  # a real status, just not 2xx
            try:
                text = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                text = str(exc)
            return HttpResponse(text=text, status=int(getattr(exc, "code", 0) or 0))
        except Exception:  # noqa: BLE001 — DNS/timeout/connection refused/etc.
            return HttpResponse(text="", status=0)

    def get(self, url: str) -> HttpResponse:
        return self._request(url, data=None)

    def post(self, url: str, data: dict[str, Any] | None = None) -> HttpResponse:
        body = urllib.parse.urlencode(data or {}).encode("utf-8")
        return self._request(url, data=body)


def _strip_html(text: str) -> str:
    """Best-effort HTML → readable text.

    The WebShop server renders HTML; the policy only needs the instruction + the
    visible button/item labels.  A real BeautifulSoup parse would be nicer but
    would add a dependency — a tag-strip + whitespace-collapse keeps the actionable
    text (button labels survive as their inner text) with zero deps.  A server that
    returns plain text or JSON passes through essentially unchanged.
    """
    if not text:
        return ""
    # Drop script/style blocks wholesale, then strip remaining tags.
    no_blocks = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    no_tags = re.sub(r"<[^>]+>", " ", no_blocks)
    # Unescape the few entities WebShop pages actually emit.
    for ent, ch in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")):
        no_tags = no_tags.replace(ent, ch)
    return re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]+", " ", no_tags)).strip()


def _coerce_page(resp: HttpResponse) -> tuple[str, float | None, bool]:
    """Normalise a server response into ``(observation_text, reward, done)``.

    The WebShop dev server has two flavours of response and we accept both:

    * **JSON** ``{"observation"|"obs": str, "reward": float, "done": bool, ...}`` —
      the shape the gym-style ``web_agent_site`` step endpoint returns and what the
      injected test fake emits.  Read those fields directly.
    * **HTML** — the human web UI.  Strip to text; reward/done are not in the body,
      so they stay ``(None, False)`` and the *caller* decides terminality (a
      ``click[buy now]`` is the terminal action — see :meth:`step`).
    """
    raw = resp.text or ""
    stripped = raw.strip()
    if stripped[:1] in ("{", "["):
        try:
            payload = json.loads(stripped)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict):
            obs = payload.get("observation")
            if obs is None:
                obs = payload.get("obs")
            if obs is None:
                obs = payload.get("text", "")
            reward = payload.get("reward")
            try:
                reward = float(reward) if reward is not None else None
            except (TypeError, ValueError):
                reward = None
            done = bool(payload.get("done", False))
            return str(obs), reward, done
    # HTML / plain text fallback.
    return _strip_html(raw), None, False


def _clip(text: str, limit: int = _MAX_OBS_CHARS) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


_SYSTEM_PROMPT = (
    "You are shopping on WebShop. Read the Instruction, then act ONE step at a "
    "time using exactly this grammar:\n"
    "  search[<query>]   - search the catalog for matching products\n"
    "  click[<target>]   - click a button on the current page: an item id, an "
    "option value (e.g. click[red]), a navigation button (click[next >], "
    "click[< prev], click[back to search]), click[description], click[features], "
    "or the terminal click[buy now].\n"
    "Search first, open an item, choose the required options, then click[buy now] "
    "to purchase the item that best matches the Instruction."
)


class WebShopEnv(AgenticEnv):
    """Real WebShop agentic environment (search/click → matching-score reward).

    Talks to the WebShop server at ``WEBSHOP_URL`` over an injectable HTTP seam.
    One episode == up to :attr:`max_turns` search/click actions ending in
    ``click[buy now]`` (or turn exhaustion).  Fail-soft throughout: an unreachable
    server, a non-2xx response, or a malformed action degrades to an observation —
    no method raises.
    """

    #: WebShop episodes are short web-navigation rollouts; 15 turns is the
    #: conventional cap (a few searches + clicks + the buy).
    max_turns: int = 15

    def __init__(
        self,
        *,
        base_url: str | None = None,
        client: HttpClient | None = None,
        http_get: Callable[[str], HttpResponse] | None = None,
        http_post: Callable[[str, dict[str, Any] | None], HttpResponse] | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        num_goals: int | None = None,
    ) -> None:
        super().__init__()
        # Resolve the server URL: explicit arg > WEBSHOP_URL env var (trailing /
        # trimmed so we can join paths uniformly).
        url = base_url if base_url is not None else os.environ.get("WEBSHOP_URL", "")
        self.base_url = (url or "").strip().rstrip("/")

        # Transport seam.  Priority: explicit get/post callables > a client object >
        # the real UrllibClient.  The callables let a test inject two plain
        # functions; the client object mirrors env_cache's injected-seam style.
        self._client: HttpClient | None = client
        self._http_get = http_get
        self._http_post = http_post
        self._timeout_s = float(timeout_s)
        if self._client is None and self._http_get is None and self._http_post is None:
            self._client = UrllibClient(timeout_s=self._timeout_s)

        #: Total goals the server exposes (for deterministic task→goal mapping).
        self._num_goals = int(num_goals) if num_goals else None

        # Per-episode state.
        self._session_id: str = ""
        self._goal_idx: int = 0
        self._last_obs: str = ""
        self._unavailable_reason: str | None = None

    # --- HTTP seam dispatch --------------------------------------------------

    def _get(self, path: str) -> HttpResponse:
        url = self._url(path)
        try:
            if self._http_get is not None:
                return self._http_get(url)
            if self._client is not None:
                return self._client.get(url)
        except Exception:  # noqa: BLE001 — a buggy injected seam must not crash the cell
            return HttpResponse(text="", status=0)
        return HttpResponse(text="", status=0)

    def _post(self, path: str, data: dict[str, Any] | None = None) -> HttpResponse:
        url = self._url(path)
        try:
            if self._http_post is not None:
                return self._http_post(url, data)
            if self._client is not None:
                return self._client.post(url, data)
        except Exception:  # noqa: BLE001
            return HttpResponse(text="", status=0)
        return HttpResponse(text="", status=0)

    def _url(self, path: str) -> str:
        path = path if path.startswith("/") else "/" + path
        return f"{self.base_url}{path}"

    # --- availability --------------------------------------------------------

    @classmethod
    def available(cls, *, client: HttpClient | None = None, base_url: str | None = None) -> tuple[bool, str]:
        """Report ``(ok, reason)`` for whether the WebShop server is reachable.

        ``WEBSHOP_URL`` must be set (or ``base_url`` passed) and a quick GET of the
        root must succeed.  Never raises — a failed probe is reported as
        ``(False, reason)``.  ``client`` is injectable so the test can assert the
        unset-URL branch without any network.
        """
        url = (base_url if base_url is not None else os.environ.get("WEBSHOP_URL", "")) or ""
        url = url.strip().rstrip("/")
        if not url:
            return False, "WEBSHOP_URL is not set"
        probe = client or UrllibClient(timeout_s=min(_DEFAULT_TIMEOUT_S, 4.0))
        try:
            resp = probe.get(url + "/")
        except Exception as exc:  # noqa: BLE001
            return False, f"WebShop probe raised {type(exc).__name__}: {exc}"
        if getattr(resp, "ok", False):
            return True, f"WebShop reachable at {url}"
        return False, f"WebShop probe to {url} returned status {getattr(resp, 'status', 0)}"

    def _unavailable_state(self) -> str | None:
        """Return a non-empty reason iff this instance's server is unreachable.

        The single source of truth both :meth:`reset` and :meth:`step` consult so
        they agree on availability.  Unavailable means either no URL was ever
        configured, OR the landing fetch in :meth:`reset` came back unreachable
        (status 0 / non-2xx) and recorded ``_unavailable_reason``.  When set, the
        first :meth:`step` returns the same terminal ``unavailable`` result the
        empty-URL path returns instead of burning turns on degraded pages.
        """
        if not self.base_url:
            return self._unavailable_reason or "WEBSHOP_URL is not set"
        if self._unavailable_reason:
            return self._unavailable_reason
        return None

    # --- action parsing ------------------------------------------------------

    @staticmethod
    def _parse_action(action: str) -> tuple[str | None, str]:
        """Parse one model turn into ``(verb, argument)``.

        Forgiving by design — a live policy emits noisy text.  Accepts:
        ``search[red shoes]`` / ``Search[ red shoes ]`` / ``search: red shoes`` /
        ``click[Buy Now]`` / fenced `````search[x]````` /
        an action embedded in prose.  Returns ``(None, "")`` when nothing parses, so
        the caller can nudge the policy.
        """
        if not action:
            return None, ""
        text = str(action).strip()
        # Strip Markdown code fences and surrounding backticks.
        text = text.replace("```", " ").replace("`", " ").strip()

        # Preferred form: verb[ ... ].  Search anywhere in the string and take the
        # LAST match (models often narrate then act: "I will search[x]").
        bracket = list(re.finditer(r"(search|click)\s*\[\s*(.*?)\s*\]", text, flags=re.I | re.S))
        if bracket:
            m = bracket[-1]
            return m.group(1).lower(), m.group(2).strip()

        # Colon / bare form: "search: red shoes" or a leading "click foo".
        colon = re.search(r"\b(search|click)\b\s*[:=]?\s*(.+)", text, flags=re.I | re.S)
        if colon:
            verb = colon.group(1).lower()
            arg = colon.group(2).strip()
            # Take only the first line of the argument (drop trailing rationale).
            arg = arg.splitlines()[0].strip() if arg else ""
            # Strip wrapping quotes the model may add.
            arg = arg.strip().strip('"').strip("'").strip()
            return verb, arg

        return None, ""

    @staticmethod
    def _is_buy(target: str) -> bool:
        """True if the click target is the terminal purchase button."""
        t = re.sub(r"[^a-z]+", " ", (target or "").lower()).strip()
        return t in {"buy now", "buy", "purchase", "place order", "buy it now"}

    @staticmethod
    def _coerce_seed(seed: Any) -> int:
        """Coerce any ``seed`` to a non-negative int — never raises.

        ``reset`` / ``_resolve_goal_idx`` derive the session id and goal index from
        the seed, so a garbage seed (a non-numeric string, a float, ``None``) must
        not propagate an exception into the rollout (the fail-soft contract).
        Tries ``int(seed)`` first (handles ints, numeric strings, floats); on
        failure falls back to a deterministic ``0`` so the same bad seed always
        maps to the same episode.
        """
        if seed is None:
            return 0
        try:
            return int(seed)
        except (TypeError, ValueError):
            return 0

    # --- episode lifecycle ---------------------------------------------------

    def reset(self, *, seed: int | None = None, task: Any = None) -> str:
        """Start a new shopping episode and return the initial page observation.

        ``task`` selects a goal/instruction index on the server.  Accepts an int,
        a dict (``{"goal_idx": n}`` / ``{"index": n}`` / ``{"task": n}``), or a
        numeric string; anything else maps deterministically via ``seed``.  The
        episode session id is derived from ``(goal_idx, seed)`` so the same cell
        replays the same instruction (the determinism contract).
        """
        self._goal_idx = self._resolve_goal_idx(task=task, seed=seed)
        seed_part = self._coerce_seed(seed)
        # Deterministic, server-friendly session id (alnum + underscore only).
        self._session_id = f"sdar_{self._goal_idx}_{seed_part}"
        self._unavailable_reason = None
        self._start_episode(system=_SYSTEM_PROMPT)

        # Fetch the instruction / search-box landing page for this session.
        obs = self._fetch_landing()
        self._last_obs = obs
        self._record_obs(obs)
        return obs

    def _resolve_goal_idx(self, *, task: Any, seed: int | None) -> int:
        """Map a task descriptor to a non-negative goal index (deterministic)."""
        idx: int | None = None
        if isinstance(task, bool):  # bool is an int subclass — treat as no-task
            idx = None
        elif isinstance(task, int):
            idx = task
        elif isinstance(task, dict):
            for key in ("goal_idx", "index", "idx", "task", "instruction_idx", "session"):
                if key in task:
                    try:
                        idx = int(task[key])
                    except (TypeError, ValueError):
                        idx = None
                    break
        elif isinstance(task, str) and task.strip().lstrip("-").isdigit():
            idx = int(task.strip())

        if idx is None:
            idx = self._coerce_seed(seed)
        if idx < 0:
            idx = -idx
        if self._num_goals and self._num_goals > 0:
            idx %= self._num_goals
        return idx

    def _fetch_landing(self) -> str:
        """GET the per-session instruction/search page; degrade on failure."""
        if not self.base_url:
            self._unavailable_reason = "WEBSHOP_URL is not set"
            return (
                "[WebShop unavailable] No server URL configured. "
                "Emit search[<query>] once the server is up."
            )
        # Canonical web_agent_site landing route is "/<session_id>"; the dev/gym
        # server also accepts a goal index query.  Either response is coerced.
        resp = self._get(f"/{urllib.parse.quote(self._session_id)}?goal_idx={self._goal_idx}")
        if not getattr(resp, "ok", False):
            self._unavailable_reason = (
                f"landing GET returned status {getattr(resp, 'status', 0)}"
            )
            return (
                "[WebShop degraded] Could not load the instruction page "
                f"(status {getattr(resp, 'status', 0)}). Try search[<query>]."
            )
        obs, _reward, _done = _coerce_page(resp)
        return _clip(obs) if obs else "[WebShop] Instruction page was empty. Try search[<query>]."

    def step(self, action: str) -> StepResult:
        """Apply one search/click action; return the resulting :class:`StepResult`.

        Never raises.  Flow:

        1. Record the raw action (counts the turn).
        2. If the server is unavailable (no URL, or reset()'s landing came back
           unreachable — see :meth:`_unavailable_state`), finish immediately with
           a zero-reward terminal step carrying ``info={"unavailable": True, ...}``.
        3. Parse the action.  Unparseable / unsupported → a nudge observation, not
           done (wastes the turn).
        4. ``search[...]`` → POST the query, observe the results page.
        5. ``click[buy now]`` → POST the purchase, read the matching reward, finish.
        6. ``click[...]`` (non-terminal) → POST the click, observe the new page; if
           the server itself signals ``done`` with a reward, honour it.
        7. On the last allowed turn without a purchase → finish with reward 0.0.
        """
        self._record_act(action)

        # (2) Unavailable server → terminal, fail-soft.  Covers BOTH no URL at all
        # and a set-but-dead URL whose landing fetch came back unreachable in
        # reset() (status 0 / non-2xx).  Without this second case a dead server
        # would burn every turn on status-0 degraded pages instead of returning
        # the promised unavailable terminal.
        reason = self._unavailable_state()
        if reason is not None:
            return self._terminal_unavailable(reason)

        verb, arg = self._parse_action(action)

        # (3) Could not parse a valid action → nudge, do not consume terminality.
        if verb is None:
            obs = (
                "Invalid action. Use search[<query>] or click[<target>] "
                "(e.g. search[red running shoes], click[item id], click[buy now])."
            )
            return self._observe(obs, done=False)

        if verb == _SEARCH:
            return self._do_search(arg)

        # verb == _CLICK
        if self._is_buy(arg):
            return self._do_buy(arg)
        return self._do_click(arg)

    # --- action handlers -----------------------------------------------------

    def _do_search(self, query: str) -> StepResult:
        if not query:
            return self._observe(
                "Empty search. Use search[<query>] with one or more keywords.", done=False
            )
        resp = self._post(
            f"/search_results/{urllib.parse.quote(self._session_id)}",
            {"search_query": query, "keywords": query, "page": 1},
        )
        obs, reward, done = self._page(resp, fallback=f"No results page returned for '{query}'.")
        # A search never ends an episode on its own, but honour an explicit server
        # terminal+reward just in case (defensive).
        if done and reward is not None:
            return self._finish_with(reward, obs)
        return self._observe(obs, done=self._exhausted())

    def _do_click(self, target: str) -> StepResult:
        if not target:
            return self._observe(
                "Empty click. Use click[<target>] (an item id, option, or button).",
                done=False,
            )
        resp = self._post(
            f"/click/{urllib.parse.quote(self._session_id)}",
            {"button": target, "click": target, "session_id": self._session_id},
        )
        obs, reward, done = self._page(resp, fallback=f"Clicking '{target}' returned no page.")
        # The server may end the episode on a click it deems terminal (e.g. a
        # 'Buy Now' rendered with a different label) — honour its reward.
        if done:
            return self._finish_with(reward if reward is not None else 0.0, obs)
        return self._observe(obs, done=self._exhausted())

    def _do_buy(self, target: str) -> StepResult:
        """Terminal purchase: POST the buy, read the matching-score reward."""
        resp = self._post(
            f"/click/{urllib.parse.quote(self._session_id)}",
            {"button": target or "Buy Now", "click": target or "Buy Now",
             "done": True, "session_id": self._session_id},
        )
        obs, reward, _done = self._page(resp, fallback="Purchase submitted.")
        if reward is None:
            # The HTML buy page does not embed a score; fall back to a dedicated
            # /done route that the gym server exposes, else 0.0.
            done_resp = self._get(f"/done/{urllib.parse.quote(self._session_id)}")
            d_obs, d_reward, _ = self._page(done_resp, fallback=obs or "Purchase complete.")
            reward = d_reward if d_reward is not None else 0.0
            if d_obs:
                obs = d_obs
        return self._finish_with(reward, obs)

    # --- observation / terminal helpers --------------------------------------

    def _page(self, resp: HttpResponse, *, fallback: str) -> tuple[str, float | None, bool]:
        """Coerce a response into ``(obs, reward, done)`` with a degraded fallback."""
        if not getattr(resp, "ok", False):
            return (
                f"[WebShop degraded] Server returned status {getattr(resp, 'status', 0)}. "
                "Try a different action.",
                None,
                False,
            )
        obs, reward, done = _coerce_page(resp)
        if not obs:
            obs = fallback
        return _clip(obs), reward, done

    def _observe(self, obs: str, *, done: bool) -> StepResult:
        """Record a non-terminal (or turn-exhausted) observation and return it."""
        self._last_obs = obs
        self._record_obs(obs)
        if done:
            # Turn budget exhausted without a purchase → zero-reward terminal.
            return self._finish_with(0.0, obs)
        return StepResult(observation=obs, reward=0.0, done=False)

    def _finish_with(self, reward: float, obs: str) -> StepResult:
        """Record the final observation, set the terminal reward, return done=True."""
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = 0.0
        reward = max(0.0, min(1.0, reward))  # WebShop matching score is in [0, 1]
        self._last_obs = obs
        self._record_obs(obs)
        info = {
            "success": reward > 0.5,
            "reward": reward,
            "steps": self.turns_taken,
        }
        self._finish(reward, info=info)
        return StepResult(observation=obs, reward=reward, done=True, info=info)

    def _terminal_unavailable(self, reason: str) -> StepResult:
        """First-step fail-soft when the server is unreachable (no raise)."""
        obs = f"[WebShop unavailable] {reason}"
        self._last_obs = obs
        self._record_obs(obs)
        info = {"unavailable": True, "reason": reason, "reward": 0.0, "steps": self.turns_taken}
        self._finish(0.0, info=info)
        return StepResult(observation=obs, reward=0.0, done=True, info=info)

    def _exhausted(self) -> bool:
        """True once the policy has used its whole turn budget without buying."""
        return self.turns_taken >= self.max_turns
