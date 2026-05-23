from backend.agents.rlm.primitives import extract_hyperparameters, _HINT_THRESHOLD

_SHORT_SLICE = "Trained with Adam, learning rate 3e-4, batch size 64, for 200 epochs."


def test_extract_hyperparameters_flat_dict(make_context, tmp_path):
    ctx = make_context(tmp_path)
    result = extract_hyperparameters(_SHORT_SLICE, ctx=ctx)
    # Semantic keys always present
    assert "optimizer" in result
    assert "learning_rate" in result
    assert "batch_size" in result
    assert "epochs_or_steps" in result
    # Assert the extracted CONTENT, not just non-emptiness. If the heuristic
    # captures the value with surrounding text, narrow these to the real
    # output — but they must check the extracted value, never `x or x`.
    assert "3e-4" in result["learning_rate"]
    assert "64" in result["batch_size"]
    assert "adam" in result["optimizer"].lower()
    assert "200" in result["epochs_or_steps"]


def test_extract_hyperparameters_no_meta_for_short_slice(make_context, tmp_path):
    """_meta must be absent when the slice is below the threshold."""
    ctx = make_context(tmp_path)
    assert len(_SHORT_SLICE) < _HINT_THRESHOLD
    result = extract_hyperparameters(_SHORT_SLICE, ctx=ctx)
    assert "_meta" not in result


def test_extract_hyperparameters_meta_hint_for_large_slice(make_context, tmp_path):
    """_meta.hint must be present when the slice exceeds the threshold."""
    ctx = make_context(tmp_path)
    large_slice = "y" * (_HINT_THRESHOLD + 1)
    result = extract_hyperparameters(large_slice, ctx=ctx)
    assert "_meta" in result
    meta = result["_meta"]
    assert "hint" in meta
    assert "rlm_query" in meta["hint"]
    assert meta["slice_chars"] == len(large_slice)
    assert meta["threshold"] == _HINT_THRESHOLD
