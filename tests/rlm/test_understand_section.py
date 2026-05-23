from backend.agents.rlm.primitives import understand_section, _HINT_THRESHOLD

SLICE = (
    "We train with the Adam optimizer at learning rate 3e-4, batch size 64, "
    "for 200 epochs. We evaluate on the CartPole-v1 dataset and report mean "
    "reward and success rate."
)


def test_understand_section_returns_partial_claim_map(make_context, tmp_path):
    ctx = make_context(tmp_path)
    result = understand_section(SLICE, ctx=ctx)
    assert isinstance(result["datasets"], list)
    assert result["training_recipe"]["optimizer"]  # Adam was detected
    assert any(d["name"] == "CartPole-v1" for d in result["datasets"])


def test_understand_section_no_meta_for_short_slice(make_context, tmp_path):
    """_meta must be absent when the slice is below the threshold."""
    ctx = make_context(tmp_path)
    assert len(SLICE) < _HINT_THRESHOLD
    result = understand_section(SLICE, ctx=ctx)
    assert "_meta" not in result


def test_understand_section_meta_hint_for_large_slice(make_context, tmp_path):
    """_meta.hint must be present when the slice exceeds the threshold."""
    ctx = make_context(tmp_path)
    large_slice = "x" * (_HINT_THRESHOLD + 1)
    result = understand_section(large_slice, ctx=ctx)
    assert "_meta" in result
    meta = result["_meta"]
    assert "hint" in meta
    assert "rlm_query" in meta["hint"]
    assert meta["slice_chars"] == len(large_slice)
    assert meta["threshold"] == _HINT_THRESHOLD
