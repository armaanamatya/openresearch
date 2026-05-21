from backend.agents.rlm.primitives import extract_hyperparameters


def test_extract_hyperparameters_flat_dict(make_context, tmp_path):
    ctx = make_context(tmp_path)
    result = extract_hyperparameters(
        "Trained with Adam, learning rate 3e-4, batch size 64, for 200 epochs.",
        ctx=ctx,
    )
    assert set(result) == {"optimizer", "learning_rate", "batch_size",
                           "epochs_or_steps", "scheduler", "other_hparams"}
    # Assert the extracted CONTENT, not just non-emptiness. If the heuristic
    # captures the value with surrounding text, narrow these to the real
    # output — but they must check the extracted value, never `x or x`.
    assert "3e-4" in result["learning_rate"]
    assert "64" in result["batch_size"]
    assert "adam" in result["optimizer"].lower()
