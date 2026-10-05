"""
Automated Unit Tests for Landmark Normalization & Invariance
Matches Acceptance Criteria for B1 in Prompt Library.
"""

import os
import json
import numpy as np
import pytest
from training.normalize import (
    normalize_hand_landmarks,
    normalize_dual_hands,
    normalize_numpy
)


def test_test_vectors_parity():
    """Verify normalization matches test_vectors.json within 1e-4."""
    vectors_path = os.path.join("training", "test_vectors.json")
    assert os.path.exists(vectors_path), "test_vectors.json must exist"

    with open(vectors_path, "r", encoding="utf-8") as f:
        test_vectors = json.load(f)

    for item in test_vectors:
        hands = item["hands"]
        expected = item["expected_normalized"]
        actual = normalize_dual_hands(hands)

        np.testing.assert_allclose(
            actual, expected,
            atol=1e-4,
            err_msg=f"Test vector {item['test_index']} failed parity check"
        )


def test_translation_invariance():
    """Verify that shifting hand in (x, y, z) does not change normalized output."""
    lms = []
    # Base hand
    wrist = {"x": 0.5, "y": 0.5, "z": 0.0}
    lms.append(wrist)
    for i in range(1, 21):
        lms.append({"x": 0.5 + i * 0.01, "y": 0.5 - i * 0.01, "z": i * 0.005})

    norm_base = normalize_hand_landmarks(lms)

    # Shifted hand by (+0.25, -0.30, +0.10)
    shifted_lms = [
        {"x": pt["x"] + 0.25, "y": pt["y"] - 0.30, "z": pt["z"] + 0.10}
        for pt in lms
    ]
    norm_shifted = normalize_hand_landmarks(shifted_lms)

    np.testing.assert_allclose(
        norm_shifted, norm_base,
        atol=1e-5,
        err_msg="Normalization must be translation invariant"
    )


def test_scale_invariance():
    """Verify that scaling hand does not change normalized output."""
    lms = []
    wrist = {"x": 0.4, "y": 0.6, "z": 0.0}
    lms.append(wrist)
    for i in range(1, 21):
        lms.append({"x": 0.4 + i * 0.015, "y": 0.6 - i * 0.02, "z": i * 0.002})

    norm_base = normalize_hand_landmarks(lms)

    # Scaled by 1.8x around wrist
    scaled_lms = [wrist]
    for pt in lms[1:]:
        scaled_lms.append({
            "x": wrist["x"] + (pt["x"] - wrist["x"]) * 1.8,
            "y": wrist["y"] + (pt["y"] - wrist["y"]) * 1.8,
            "z": wrist["z"] + (pt["z"] - wrist["z"]) * 1.8
        })

    norm_scaled = normalize_hand_landmarks(scaled_lms)

    np.testing.assert_allclose(
        norm_scaled, norm_base,
        atol=1e-5,
        err_msg="Normalization must be scale invariant"
    )


def test_missing_hand_zero_padding():
    """Verify that a missing hand yields 63 zeros in its slot."""
    one_hand = [{
        "label": "Right",
        "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    }]
    vec = normalize_dual_hands(one_hand)
    assert len(vec) == 126
    # Left hand slot (0..62) must be all zeros
    assert all(v == 0.0 for v in vec[:63])
