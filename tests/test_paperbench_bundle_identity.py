from pathlib import Path


PAPERBENCH_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "paperbench"

EXPECTED_TITLES = {
    "ftrl": "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem",
    "mechanistic-understanding": "A Mechanistic Understanding of Alignment Algorithms",
    "sequential-neural-score-estimation": "Sequential Neural Score Estimation",
}


def test_vendored_paperbench_bundle_identities() -> None:
    for bundle_id, expected_title in EXPECTED_TITLES.items():
        bundle_dir = PAPERBENCH_ROOT / bundle_id
        assert (bundle_dir / "paper.md").is_file()
        assert (bundle_dir / "rubric.json").is_file()
        first_chunk = (bundle_dir / "paper.md").read_text(encoding="utf-8")[:200]
        assert expected_title in first_chunk
