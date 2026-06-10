"""Tests for check_user_messages and respond_to_user primitives."""

from __future__ import annotations

import json


from backend.agents.rlm.primitives import check_user_messages, respond_to_user


# ---------------------------------------------------------------------------
# check_user_messages
# ---------------------------------------------------------------------------

class TestCheckUserMessages:
    def test_returns_empty_when_no_file(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        result = check_user_messages(ctx=ctx)
        assert result == []

    def test_returns_empty_when_file_is_empty(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        (ctx.project_dir / "user_messages.jsonl").write_text("")
        result = check_user_messages(ctx=ctx)
        assert result == []

    def test_returns_new_user_messages(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        msgs_path = ctx.project_dir / "user_messages.jsonl"
        msgs_path.write_text(
            json.dumps({"role": "user", "content": "hello", "ts": "2026-01-01T00:00:00+00:00"}) + "\n"
        )
        result = check_user_messages(ctx=ctx)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_does_not_return_assistant_messages(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        msgs_path = ctx.project_dir / "user_messages.jsonl"
        msgs_path.write_text(
            json.dumps({"role": "assistant", "content": "reply", "ts": "t"}) + "\n"
        )
        result = check_user_messages(ctx=ctx)
        assert result == []

    def test_cursor_advances_on_first_call(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        msgs_path = ctx.project_dir / "user_messages.jsonl"
        msgs_path.write_text(
            json.dumps({"role": "user", "content": "first", "ts": "t"}) + "\n"
        )
        # First call reads the message and advances the cursor
        result1 = check_user_messages(ctx=ctx)
        assert len(result1) == 1

        # Second call with no new lines returns empty
        result2 = check_user_messages(ctx=ctx)
        assert result2 == []

    def test_cursor_returns_only_new_messages(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        msgs_path = ctx.project_dir / "user_messages.jsonl"
        msgs_path.write_text(
            json.dumps({"role": "user", "content": "first", "ts": "t"}) + "\n"
        )
        # First call consumes "first"
        check_user_messages(ctx=ctx)

        # Append a second message
        with msgs_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"role": "user", "content": "second", "ts": "t"}) + "\n")

        # Second call returns only "second"
        result2 = check_user_messages(ctx=ctx)
        assert len(result2) == 1
        assert result2[0]["content"] == "second"

    def test_cursor_file_written(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        msgs_path = ctx.project_dir / "user_messages.jsonl"
        msgs_path.write_text(
            json.dumps({"role": "user", "content": "x", "ts": "t"}) + "\n"
        )
        check_user_messages(ctx=ctx)
        cursor_path = ctx.project_dir / "_user_message_cursor.json"
        assert cursor_path.exists()
        data = json.loads(cursor_path.read_text())
        assert data["offset"] == 1  # one line was read

    def test_bad_cursor_file_resets_to_zero(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        cursor_path = ctx.project_dir / "_user_message_cursor.json"
        cursor_path.write_text("not-json")
        msgs_path = ctx.project_dir / "user_messages.jsonl"
        msgs_path.write_text(
            json.dumps({"role": "user", "content": "hello", "ts": "t"}) + "\n"
        )
        # Bad cursor → resets to 0 → reads from beginning
        result = check_user_messages(ctx=ctx)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# respond_to_user
# ---------------------------------------------------------------------------

class TestRespondToUser:
    def test_returns_sent_true(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        result = respond_to_user("Here is my reply.", ctx=ctx)
        assert result["sent"] is True
        assert result["outcome"] == "ok"

    def test_appends_to_user_messages_jsonl(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        respond_to_user("my reply", ctx=ctx)
        path = ctx.project_dir / "user_messages.jsonl"
        assert path.exists()
        entry = json.loads(path.read_text().strip())
        assert entry["role"] == "assistant"
        assert entry["content"] == "my reply"
        assert "ts" in entry

    def test_appends_dashboard_event(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        respond_to_user("my reply", ctx=ctx)
        path = ctx.project_dir / "dashboard_events.jsonl"
        assert path.exists()
        entry = json.loads(path.read_text().strip())
        assert entry["event"] == "user_message_response"
        assert entry["content"] == "my reply"

    def test_empty_message_returns_error(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        result = respond_to_user("", ctx=ctx)
        assert result["sent"] is False
        assert "error" in result

    def test_whitespace_message_returns_error(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        result = respond_to_user("   ", ctx=ctx)
        assert result["sent"] is False

    def test_does_not_raise_on_empty_message(self, make_context, tmp_path):
        """respond_to_user must be fail-soft — never raise."""
        ctx = make_context(tmp_path)
        # Should not raise; returns error dict instead
        result = respond_to_user("", ctx=ctx)
        assert isinstance(result, dict)

    def test_multiple_replies_accumulate(self, make_context, tmp_path):
        ctx = make_context(tmp_path)
        respond_to_user("reply one", ctx=ctx)
        respond_to_user("reply two", ctx=ctx)
        lines = (ctx.project_dir / "user_messages.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["content"] == "reply two"
