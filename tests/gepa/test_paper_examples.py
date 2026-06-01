"""PaperExamplesBuilder constructs training examples from understand_section outputs."""
from backend.agents.gepa.trainset.paper_examples import PaperExamplesBuilder, PaperExample


UNDERSTAND_OUTPUT = {
    "datasets": [{"name": "CIFAR-10", "split": "train", "size": 50000}],
    "metrics": [{"name": "top1_accuracy", "direction": "higher", "bounds": [0, 1]}],
    "training_recipe": {"optimizer": "Adam", "learning_rate": 0.001, "batch_size": 64, "epochs": 100},
    "hardware_clues": ["GPU required", "16GB VRAM"],
    "ambiguities": [],
    "outcome": "ok",
}

EXTRACT_OUTPUT = {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs_or_steps": 100,
    "scheduler": None,
    "other_hparams": {"weight_decay": 1e-4},
}

ENV_SPEC = {
    "base_image": "pytorch/pytorch:2.1.0",
    "requirements": ["torch", "torchvision"],
}


def test_build_returns_list_of_paper_examples():
    builder = PaperExamplesBuilder(
        understand_output=UNDERSTAND_OUTPUT,
        extract_output=EXTRACT_OUTPUT,
        env_spec=ENV_SPEC,
        expected_metric_ids=["top1_accuracy"],
    )
    examples = builder.build(n=3)
    assert len(examples) >= 1
    assert all(isinstance(e, PaperExample) for e in examples)


def test_paper_example_has_required_fields():
    builder = PaperExamplesBuilder(
        understand_output=UNDERSTAND_OUTPUT,
        extract_output=EXTRACT_OUTPUT,
        env_spec=ENV_SPEC,
        expected_metric_ids=["top1_accuracy"],
    )
    example = builder.build(n=1)[0]
    assert isinstance(example.method_spec, dict)
    assert isinstance(example.env_spec, dict)
    assert isinstance(example.expected_metric_ids, list)
    assert "top1_accuracy" in example.expected_metric_ids


def test_build_with_no_understand_output():
    builder = PaperExamplesBuilder(
        understand_output=None,
        extract_output=None,
        env_spec=ENV_SPEC,
        expected_metric_ids=[],
    )
    examples = builder.build(n=1)
    assert len(examples) >= 1  # falls back to minimal example
