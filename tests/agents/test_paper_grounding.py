"""PR-ξ Phase 3 — paper_grounding module tests.

Tests that SDAR-contaminated method_specs are flagged when paired with
non-SDAR paper text, while correctly-grounded claims pass.
"""
from __future__ import annotations

from pathlib import Path


from backend.agents.paper_grounding import assert_paper_grounded

# ---------------------------------------------------------------------------
# VAE paper text fixture
# ---------------------------------------------------------------------------

_VAE_PAPER_PATH = Path(
    "/home/abheekp/openresearch/runs/"
    "_preserved_vae_score_0.6457_prj_03271ba130d423fe/parsed_full_text.txt"
)

# Representative fallback text with Frey Face (used when the actual file is absent)
_VAE_FALLBACK_TEXT = (
    "Auto-Encoding Variational Bayes. We evaluated our method on two image datasets: "
    "the Frey Face dataset of 1965 images (28x20 pixels each) and the MNIST digits dataset. "
    "We used the AEVB algorithm with variational auto-encoders. "
    "Our method converged faster than the wake-sleep baseline."
)


def _get_vae_paper_text() -> str:
    if _VAE_PAPER_PATH.exists():
        return _VAE_PAPER_PATH.read_text(encoding="utf-8", errors="replace")
    return _VAE_FALLBACK_TEXT


# ---------------------------------------------------------------------------
# 1. Grounded claim passes
# ---------------------------------------------------------------------------

class TestGroundedClaimPasses:
    def test_grounded_claim_passes(self):
        """A claim referencing datasets that appear in the paper text must pass."""
        paper_text = (
            "We evaluate on the MNIST handwritten digit dataset using a variational autoencoder "
            "with amortised inference. Each image is 28x28 pixels, greyscale. "
            "We report the average test log-likelihood."
        )
        claim_map = {
            "core_contribution": "variational autoencoder with amortised inference",
            "datasets": ["MNIST"],
            "claims": [
                {"dataset": "MNIST", "metric": "log-likelihood", "method": "variational autoencoder"},
            ],
        }
        violations = assert_paper_grounded(claim_map, paper_text)
        assert not violations, (
            f"Expected no violations for MNIST in MNIST-mentioning text, got: {violations}"
        )


# ---------------------------------------------------------------------------
# 2. SDAR datasets in VAE paper text flagged
# ---------------------------------------------------------------------------

class TestSdarInVaePaperTextFlagged:
    def test_sdar_in_vae_paper_text_flagged(self):
        """A claim_map with SDAR datasets (ALFWorld, WebShop) should produce
        violations when the paper text is from the VAE paper."""
        paper_text = _get_vae_paper_text()
        claim_map = {
            "core_contribution": "Agentic reinforcement learning with self-distillation",
            "datasets": ["ALFWorld", "WebShop"],
            "claims": [
                {"dataset": "ALFWorld", "metric": "success_rate"},
                {"dataset": "WebShop", "metric": "reward"},
            ],
        }
        violations = assert_paper_grounded(claim_map, paper_text)
        violated_values = {v.value for v in violations}
        assert "ALFWorld" in violated_values, (
            f"Expected ALFWorld to be flagged as unfounded in VAE paper text; "
            f"violations: {violations}"
        )
        assert "WebShop" in violated_values, (
            f"Expected WebShop to be flagged as unfounded in VAE paper text; "
            f"violations: {violations}"
        )


# ---------------------------------------------------------------------------
# 3. Case-insensitive matching
# ---------------------------------------------------------------------------

class TestCaseInsensitiveMatch:
    def test_case_insensitive_match(self):
        """Dataset names should match regardless of case in the paper text."""
        paper_text = (
            "We evaluated on the frey face dataset of 1965 images "
            "collected from a video of face expressions. "
            "Our generative model learns faces effectively."
        )
        claim_map = {
            "core_contribution": "generative model for faces",
            "datasets": ["Frey Face"],
        }
        violations = assert_paper_grounded(claim_map, paper_text)
        assert not violations, (
            f"Expected no violations for 'Frey Face' in text with 'frey face', got: {violations}"
        )


# ---------------------------------------------------------------------------
# 4. Token-overlap threshold controls multi-word match
# ---------------------------------------------------------------------------

class TestTokenOverlapThreshold:
    def test_below_threshold_flagged(self):
        """With threshold=0.5, a 4-token method name with only 1 matching token
        (25% overlap) should be flagged."""
        paper_text = (
            "We propose a self-distilled training objective that reduces variance. "
            "The online policy is updated using GRPO."
        )
        claim_map = {
            "core_contribution": "policy training",
            "datasets": [],
            "claims": [
                # 4 non-stopword tokens: "self", "distilled", "online", "policy"
                # paper_text has "self-distilled" → "self" and "distilled" match
                # "online" → matches, "policy" → matches  (all match)
                # Use a name that's definitely absent
                {"method": "ALFWorld WebShop HotpotQA SearchQA", "metric": "reward"},
            ],
        }
        violations = assert_paper_grounded(claim_map, paper_text, min_overlap_threshold=0.5)
        # "ALFWorld WebShop HotpotQA SearchQA" — none of these appear in paper_text
        violated_values = {v.value for v in violations}
        assert "ALFWorld WebShop HotpotQA SearchQA" in violated_values, (
            f"Expected unfounded multi-word name to be flagged; violations: {violations}"
        )

    def test_above_threshold_passes(self):
        """With threshold=0.25, a 4-token name with 1 matching token (25%) should pass."""
        paper_text = (
            "We propose a Self-Distilled training method based on reinforcement learning."
        )
        # "Self Distilled Online Policy" — 4 tokens; "self" and "distilled" in paper_text
        # = 2/4 = 50% overlap, above threshold=0.25 AND above threshold=0.5
        # Use threshold=0.25 to test the boundary explicitly
        claim_map = {
            "core_contribution": "policy training",
            "datasets": [],
            "claims": [
                {"method": "Self Distilled Training Method", "metric": "reward"},
            ],
        }
        violations = assert_paper_grounded(claim_map, paper_text, min_overlap_threshold=0.25)
        violated_values = {v.value for v in violations}
        assert "Self Distilled Training Method" not in violated_values, (
            f"Expected grounded method not to be flagged at threshold=0.25; "
            f"violations: {violations}"
        )

    def test_strict_threshold_flags_partial_match(self):
        """With threshold=0.5, a name with only 25% token overlap is flagged."""
        paper_text = "We use reinforcement learning."
        # "Self Distilled Online Policy" — 4 key tokens, only "reinforcement" matches
        # but that's not in the name. Let's use a name where exactly 1/4 tokens match.
        # "learning online policy training": "learning" in paper_text but not others
        claim_map = {
            "core_contribution": "test",
            "datasets": [],
            "claims": [
                # 4 unique non-stopword tokens; only one matches
                {"method": "ALFWorld WebShop SearchQA HotpotQA", "metric": "score"},
            ],
        }
        violations = assert_paper_grounded(claim_map, paper_text, min_overlap_threshold=0.5)
        violated_values = {v.value for v in violations}
        assert "ALFWorld WebShop SearchQA HotpotQA" in violated_values, (
            f"Expected method with 0% token overlap to be flagged at threshold=0.5; "
            f"violations: {violations}"
        )


# ---------------------------------------------------------------------------
# 5. Frey Face in VAE paper text passes (regression: grounding works correctly)
# ---------------------------------------------------------------------------

class TestFreyFaceInVaeText:
    def test_frey_face_grounded_in_vae_paper(self):
        """Frey Face appears in the VAE paper — it must pass grounding."""
        paper_text = _get_vae_paper_text()
        claim_map = {
            "core_contribution": "variational autoencoder",
            "datasets": ["Frey Face"],
        }
        violations = assert_paper_grounded(claim_map, paper_text)
        violated_values = {v.value for v in violations}
        assert "Frey Face" not in violated_values, (
            f"Frey Face should be grounded in VAE paper text, but was flagged: "
            f"{[v for v in violations if v.value == 'Frey Face']}"
        )
