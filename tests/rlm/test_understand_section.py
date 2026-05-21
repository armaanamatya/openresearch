from backend.agents.rlm.primitives import understand_section

SLICE = (
    "We train with the Adam optimizer at learning rate 3e-4, batch size 64, "
    "for 200 epochs. We evaluate on the CartPole-v1 dataset and report mean "
    "reward and success rate."
)


def test_understand_section_returns_partial_claim_map(make_context, tmp_path):
    ctx = make_context(tmp_path)
    result = understand_section(SLICE, ctx=ctx)
    assert set(result) == {"datasets", "metrics", "training_recipe",
                           "hardware_clues", "ambiguities"}
    assert isinstance(result["datasets"], list)
    assert result["training_recipe"]["optimizer"]  # Adam was detected
    assert any(d["name"] == "CartPole-v1" for d in result["datasets"])
