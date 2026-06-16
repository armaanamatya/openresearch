"""Chat-steering injection — unread operator messages ride primitive results.

2026-06-10: both live runs ignored time-critical operator steering for hours
because the root never called check_user_messages() (the contract asks, nothing
enforces). Primitive results are the one channel the root always reads, so
binding.wrap_primitive attaches unread messages there — once per message, with
the formal check_user_messages cursor untouched.
"""

from __future__ import annotations

import json

from backend.agents.rlm.binding import wrap_primitive


def _steer(ctx, text):
    with (ctx.project_dir / "user_messages.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "user", "content": text, "ts": "t"}) + "\n")


def _wrapped_run_experiment(ctx):
    return wrap_primitive("run_experiment", lambda **kw: {"success": True}, ctx)


def test_unread_steering_attached_to_result(make_context, tmp_path):
    ctx = make_context(tmp_path)
    _steer(ctx, "lower the learning rate for deep variants")
    result = _wrapped_run_experiment(ctx)()
    assert result["operator_messages"] == ["lower the learning rate for deep variants"]
    assert "respond_to_user" in result["operator_messages_note"]


def test_each_message_injected_at_most_once(make_context, tmp_path):
    ctx = make_context(tmp_path)
    _steer(ctx, "first")
    fn = _wrapped_run_experiment(ctx)
    assert fn()["operator_messages"] == ["first"]
    assert "operator_messages" not in fn()  # cursor advanced
    _steer(ctx, "second")
    assert fn()["operator_messages"] == ["second"]  # only the new one


def test_formal_cursor_untouched(make_context, tmp_path):
    ctx = make_context(tmp_path)
    _steer(ctx, "msg")
    _wrapped_run_experiment(ctx)()
    # check_user_messages' own cursor file must not exist/advance — the formal
    # flow still surfaces the message when the root finally calls it.
    assert not (ctx.project_dir / "_user_message_cursor.json").exists()
    from backend.agents.rlm.primitives import check_user_messages
    msgs = check_user_messages(ctx=ctx)
    assert any("msg" in str(m) for m in msgs)


def test_non_injection_primitive_untouched(make_context, tmp_path):
    ctx = make_context(tmp_path)
    _steer(ctx, "msg")
    fn = wrap_primitive("understand_section", lambda **kw: {"success": True}, ctx)
    assert "operator_messages" not in fn()


def test_flag_off_disables(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_INJECT_STEERING", "0")
    ctx = make_context(tmp_path)
    _steer(ctx, "msg")
    assert "operator_messages" not in _wrapped_run_experiment(ctx)()


def test_no_messages_no_change(make_context, tmp_path):
    ctx = make_context(tmp_path)
    # propose_improvements has no arg-contract guard, so the dummy's result
    # passes through verbatim (run_experiment's guard pre-empts missing args).
    fn = wrap_primitive("propose_improvements", lambda **kw: {"success": True}, ctx)
    result = fn()
    assert "operator_messages" not in result
    assert result["success"] is True


def test_assistant_replies_not_reinjected(make_context, tmp_path):
    ctx = make_context(tmp_path)
    _steer(ctx, "user asks")
    fn = _wrapped_run_experiment(ctx)
    fn()
    # respond_to_user appends an assistant line; it must never be injected.
    with (ctx.project_dir / "user_messages.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "assistant", "content": "ack", "ts": "t"}) + "\n")
    assert "operator_messages" not in fn()
