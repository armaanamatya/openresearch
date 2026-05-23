"""Guard test for handoff P2-I10 / T22 — tesseract eng language data installed."""
import shutil
import subprocess
import pytest


def test_ocr_eng_language_data_is_installed():
    """Symptom: OCR fallback is non-functional; tesseract has no eng data.

    OcrPaperParser fails soft and the OCR test skips in environments missing
    the eng language data (handoff P2-I10 / T22). Verify: tesseract reports
    eng in its language list.
    """
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract binary not installed in this environment")
    out = subprocess.run(
        ["tesseract", "--list-langs"], capture_output=True, text=True, check=False
    )
    # tesseract writes the language list to stderr on some versions; check both.
    combined = (out.stdout or "") + "\n" + (out.stderr or "")
    assert "eng" in combined.lower(), (
        f"tesseract --list-langs did not include eng. Output:\n{combined}"
    )
