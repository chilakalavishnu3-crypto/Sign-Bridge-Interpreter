"""
SignBridge Canonical Landmark Normalization Module
Exact implementation as specified in SignBridge PRD, Prompt Library, and Architecture slides.

Formula:
  normalized_point = (current_point - wrist_point) / (wrist_to_middle_knuckle_distance)

Output:
  63 floats per hand (21 landmarks x 3 coords: x, y, z)
  126 floats for two hands (Left hand slot first, Right hand slot second)
  Zero-padded for missing hands.
"""

import json
import math
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Landmark indices
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
NUM_LANDMARKS = 21
COORDS_PER_HAND = NUM_LANDMARKS * 3  # 63
TOTAL_FEATURES = COORDS_PER_HAND * 2  # 126


def normalize_hand_landmarks(landmarks: List[Dict[str, float]]) -> List[float]:
    """
    Normalizes 21 3D landmarks for a single hand.
    landmarks: list of 21 dicts with keys 'x', 'y', 'z'
    Returns: list of 63 floats
    """
    if not landmarks or len(landmarks) < NUM_LANDMARKS:
        return [0.0] * COORDS_PER_HAND

    wrist = landmarks[WRIST_IDX]
    middle_mcp = landmarks[MIDDLE_MCP_IDX]

    wx, wy, wz = wrist['x'], wrist['y'], wrist.get('z', 0.0)
    mx, my, mz = middle_mcp['x'], middle_mcp['y'], middle_mcp.get('z', 0.0)

    # Euclidean distance from wrist to middle knuckle
    dist = math.sqrt((mx - wx) ** 2 + (my - wy) ** 2 + (mz - wz) ** 2)
    if dist < 1e-6:
        dist = 1.0

    normalized = []
    for lm in landmarks[:NUM_LANDMARKS]:
        nx = (lm['x'] - wx) / dist
        ny = (lm['y'] - wy) / dist
        nz = (lm.get('z', 0.0) - wz) / dist
        normalized.extend([round(nx, 6), round(ny, 6), round(nz, 6)])

    return normalized


def normalize_dual_hands(
    hands_data: List[Dict[str, Any]]
) -> List[float]:
    """
    Normalizes up to two hands into a 126-length feature vector.
    hands_data: list of dicts with:
      'label': 'Left' or 'Right'
      'landmarks': list of 21 dicts [{'x': float, 'y': float, 'z': float}, ...]
    Returns: list of 126 floats (Left hand slot [0:63], Right hand slot [63:126])
    """
    left_hand_features = [0.0] * COORDS_PER_HAND
    right_hand_features = [0.0] * COORDS_PER_HAND

    for hand in hands_data:
        label = hand.get('label', '')
        landmarks = hand.get('landmarks', [])
        norm = normalize_hand_landmarks(landmarks)

        if label == 'Left':
            left_hand_features = norm
        elif label == 'Right':
            right_hand_features = norm
        else:
            # If label unspecified, assign first detected to left, second to right
            if left_hand_features == [0.0] * COORDS_PER_HAND:
                left_hand_features = norm
            else:
                right_hand_features = norm

    return left_hand_features + right_hand_features


def correct_aspect(hands_data: List[Dict[str, Any]], aspect: float) -> List[Dict[str, Any]]:
    """
    MediaPipe returns x as a fraction of image WIDTH and y as a fraction of
    image HEIGHT, so on a 4:3 frame a real hand looks squashed sideways.
    Scale x (and z, which MediaPipe expresses on the x scale) by width/height
    to get isotropic coordinates before normalizing. aspect=1.0 is a no-op.
    """
    if not aspect or abs(aspect - 1.0) < 1e-6:
        return hands_data
    out = []
    for hand in hands_data:
        lms = [
            {"x": lm["x"] * aspect, "y": lm["y"], "z": lm.get("z", 0.0) * aspect}
            for lm in hand.get("landmarks", [])
        ]
        out.append({**hand, "landmarks": lms})
    return out


def normalize_numpy(hands_array: np.ndarray) -> np.ndarray:
    """
    Vectorized normalization for training data arrays.
    hands_array: shape (N, 2, 21, 3) or (2, 21, 3)
    Returns: shape (N, 126) or (126,)
    """
    single = False
    if hands_array.ndim == 3:
        hands_array = np.expand_dims(hands_array, axis=0)
        single = True

    N = hands_array.shape[0]
    out = np.zeros((N, TOTAL_FEATURES), dtype=np.float32)

    for i in range(N):
        frame = hands_array[i]  # shape (2, 21, 3)
        for h_idx in range(2):
            hand = frame[h_idx]  # shape (21, 3)
            if np.all(np.abs(hand) < 1e-6):
                continue
            wrist = hand[WRIST_IDX]
            mcp = hand[MIDDLE_MCP_IDX]
            dist = np.linalg.norm(mcp - wrist)
            if dist < 1e-6:
                dist = 1.0
            norm_hand = (hand - wrist) / dist
            offset = h_idx * COORDS_PER_HAND
            out[i, offset:offset + COORDS_PER_HAND] = norm_hand.flatten()

    return out[0] if single else out


def generate_test_vectors(out_path: str = "training/test_vectors.json"):
    """
    Generates 5 realistic sample test frames and their expected normalized output
    to verify JavaScript / TypeScript runtime parity within 1e-4.
    """
    np.random.seed(42)
    test_vectors = []

    for idx in range(5):
        left_active = idx in [0, 1, 3, 4]
        right_active = idx in [1, 2, 3, 4]

        frame_data = {"test_index": idx, "hands": []}

        if left_active:
            wrist = np.array([0.35 + idx * 0.02, 0.55 + idx * 0.01, 0.01])
            mcp = wrist + np.array([0.02, -0.15, -0.02])
            lms = [wrist]
            for j in range(1, 21):
                if j == 9:
                    lms.append(mcp)
                else:
                    offset = np.random.uniform(-0.08, 0.08, size=3)
                    offset[1] = -abs(offset[1]) - 0.05
                    lms.append(wrist + offset)
            frame_data["hands"].append({
                "label": "Left",
                "landmarks": [{"x": round(float(p[0]), 6), "y": round(float(p[1]), 6), "z": round(float(p[2]), 6)} for p in lms]
            })

        if right_active:
            wrist = np.array([0.65 - idx * 0.01, 0.52 + idx * 0.02, -0.01])
            mcp = wrist + np.array([-0.02, -0.16, 0.01])
            lms = [wrist]
            for j in range(1, 21):
                if j == 9:
                    lms.append(mcp)
                else:
                    offset = np.random.uniform(-0.08, 0.08, size=3)
                    offset[1] = -abs(offset[1]) - 0.05
                    lms.append(wrist + offset)
            frame_data["hands"].append({
                "label": "Right",
                "landmarks": [{"x": round(float(p[0]), 6), "y": round(float(p[1]), 6), "z": round(float(p[2]), 6)} for p in lms]
            })

        expected_norm = normalize_dual_hands(frame_data["hands"])
        frame_data["expected_normalized"] = expected_norm
        test_vectors.append(frame_data)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_vectors, f, indent=2)
    print(f"Generated {out_path} with 5 test vectors.")


if __name__ == "__main__":
    generate_test_vectors()
