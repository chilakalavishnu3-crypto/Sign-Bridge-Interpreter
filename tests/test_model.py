"""
Automated Unit Tests for Model Inference & Latency
Matches B2 Acceptance Criteria.
"""

import os
import time
import joblib
import numpy as np
import pytest
from training.dataset_generator import CLASSES
from training.train import load_dataset, HELD_OUT_SIGNER


@pytest.fixture(scope="module")
def loaded_model():
    model_path = os.path.join("assets", "model", "isl_model.joblib")
    assert os.path.exists(model_path), "Trained model must exist"
    return joblib.load(model_path)


def test_model_inference_latency(loaded_model):
    """Verify inference latency is under 15ms per frame for smooth 30+ FPS."""
    sample_feat = np.random.uniform(-1.0, 1.0, size=(1, 126)).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = loaded_model.predict_proba(sample_feat)

    # Timed run
    start_time = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        _ = loaded_model.predict_proba(sample_feat)
    avg_latency_ms = ((time.perf_counter() - start_time) / iterations) * 1000.0

    print(f"\nAverage Inference Latency: {avg_latency_ms:.2f} ms")
    assert avg_latency_ms < 15.0, f"Latency {avg_latency_ms:.2f}ms exceeds 15ms threshold"


def test_held_out_accuracy(loaded_model):
    """Verify held-out teammate accuracy is at least 85%."""
    _, _, X_test, y_test = load_dataset()
    assert len(X_test) > 0, "Test set must not be empty"

    pred_indices = loaded_model.predict(X_test)
    id_to_label = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}
    pred_labels = [id_to_label[i] for i in pred_indices]

    acc = np.mean(np.array(pred_labels) == y_test)
    print(f"\nModel Held-Out Accuracy: {acc * 100:.2f}%")
    assert acc >= 0.85, f"Accuracy {acc*100:.2f}% is below 85% requirement"
