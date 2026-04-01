import pytest
from speech_to_text_finetune.utils import compute_wer_cer_metrics


def test_compute_wer_cer_metrics():
    references = ["hello world", "test phrase"]
    predictions = ["hello world", "test phrases"]

    metrics = compute_wer_cer_metrics(references, predictions)

    assert "wer" in metrics
    assert "cer" in metrics
    assert metrics["wer"] >= 0.0
    assert metrics["cer"] >= 0.0
