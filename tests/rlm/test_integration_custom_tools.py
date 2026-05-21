import json

# A mock paper, offloaded as the REPL `context` variable — the root model
# slices `context`; it never receives the paper in its own prompt (the RLM premise).
MOCK_PAPER = {
    "paper_text": ("Our method trains with the Adam optimizer at learning rate "
                   "3e-4, batch size 64, for 200 epochs on CartPole-v1. " * 12),
    "paper_metadata": {"title": "Mock RL Paper"},
}


def test_primitives_are_callable_inside_the_rlm_repl(make_context, tmp_path):
    import rlm.core.rlm as rlm_core
    from rlm import RLM
    from rlm.clients.base_lm import BaseLM
    from rlm.core.types import ModelUsageSummary, UsageSummary

    from backend.agents.rlm.binding import build_custom_tools

    ctx = make_context(tmp_path)
    custom_tools = build_custom_tools(ctx)

    class ScriptedLM(BaseLM):
        def __init__(self):
            super().__init__(model_name="scripted")
            self.turns = 0

        def completion(self, prompt):
            self.turns += 1
            if self.turns == 1:
                # Slice the offloaded `context` variable and pass the slice to
                # a primitive — exercises the paper-as-variable RLM flow.
                return ("```repl\n"
                        "slice_ = context['paper_text'][:600]\n"
                        "hp = extract_hyperparameters(slice_)\n"
                        "report = {'hyperparameters': hp}\n"
                        "print(report)\n```\n")
            return "Done.\nFINAL_VAR(report)"

        async def acompletion(self, prompt):
            return self.completion(prompt)

        def _u(self):
            return ModelUsageSummary(total_calls=self.turns, total_input_tokens=0,
                                     total_output_tokens=0, total_cost=0.0)

        def get_usage_summary(self):
            return UsageSummary(model_usage_summaries={self.model_name: self._u()})

        def get_last_usage(self):
            return self._u()

    original = rlm_core.get_client
    rlm_core.get_client = lambda backend, kw: ScriptedLM()
    rlm = None
    try:
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "scripted"},
                  environment="local", max_iterations=4, custom_tools=custom_tools,
                  custom_sub_tools={})
        result = rlm.completion(MOCK_PAPER)  # MOCK_PAPER becomes the REPL `context`
    finally:
        rlm_core.get_client = original
        if rlm is not None:  # rlm is unbound if the RLM(...) constructor raised
            rlm.close()

    # The scripted run terminates with FINAL_VAR(report), and report is
    # {"hyperparameters": ...} — so a genuine FINAL_VAR resolution (not the
    # max_iterations _default_answer fallback) carries "hyperparameters".
    assert result.response
    assert "hyperparameters" in str(result.response)
    events = [json.loads(ln) for ln in
              (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines() if ln]
    # status == "ok" — proves the primitive COMPLETED, not merely that a
    # "start" event was emitted before it ran.
    names = {e["primitive"] for e in events
             if e.get("event") == "primitive_call" and e.get("status") == "ok"}
    assert "extract_hyperparameters" in names


def test_every_primitive_binds_and_heuristic_ones_run(make_context, tmp_path):
    """Every primitive binds into custom_tools and is callable; the three
    no-dependency heuristic primitives actually run through the wrapper.

    Together with the REPL test above and the Task 6-11 unit tests, this
    covers issue #59's "every primitive callable from the REPL" done-condition.
    """
    from backend.agents.rlm.binding import build_custom_tools

    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    assert len(tools) == 9  # the nine brief-§7 primitives
    for entry in tools.values():
        assert callable(entry["tool"])

    # The heuristic primitives need no monkeypatching — invoke them for real
    # through the bound custom_tools wrapper.
    us = tools["understand_section"]["tool"]("Adam, lr 3e-4, batch 64, CartPole-v1.")
    assert {"datasets", "metrics", "training_recipe"} <= set(us)
    hp = tools["extract_hyperparameters"]["tool"]("Adam optimizer, batch size 64.")
    assert "64" in hp["batch_size"]
    env = tools["detect_environment"]["tool"]({"core_contribution": "A PyTorch agent."})
    assert env["dockerfile"].startswith("FROM")

    events = [json.loads(ln) for ln in
              (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines() if ln]
    ran = {e["primitive"] for e in events if e.get("event") == "primitive_call"}
    assert {"understand_section", "extract_hyperparameters", "detect_environment"} <= ran
