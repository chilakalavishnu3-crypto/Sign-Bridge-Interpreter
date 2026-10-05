"""
SignBridge Biomechanical Dataset Synthesizer and Raw Data Generator
Generates realistic 3D hand landmark datasets for all 26 classes (25 signs + NONE)
simulating multiple signers (Signer_A, Signer_B, Signer_C) with 3D rotation,
joint articulation, scale variation, and left/right hand mirroring.
"""

import os
import json
import math
import numpy as np
from typing import Dict, List, Tuple
try:
    from training.normalize import normalize_dual_hands, normalize_numpy
except ImportError:
    from normalize import normalize_dual_hands, normalize_numpy

CLASSES = [
    "0", "1", "2", "3", "4", "5",
    "HELLO", "THANK YOU", "YES", "NO", "HELP",
    "WHERE", "WHEN", "WATER", "TOILET", "MONEY", "DOCTOR",
    "TICKET", "TRAIN", "PLATFORM",
    "PAIN", "MEDICINE", "FEVER", "HEAD", "STOMACH",
    "NONE"
]

SIGNERS = ["Signer_A", "Signer_B", "Signer_C"]

# Standard 21 MediaPipe hand joint hierarchy:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)


def build_base_hand(wrist=(0.0, 0.0, 0.0), scale=0.15) -> np.ndarray:
    """Creates a neutral resting hand landmark template (21, 3)."""
    lms = np.zeros((21, 3), dtype=np.float32)
    wx, wy, wz = wrist
    lms[0] = [wx, wy, wz]

    # Thumb
    lms[1] = [wx - 0.25 * scale, wy - 0.25 * scale, wz]
    lms[2] = [wx - 0.45 * scale, wy - 0.50 * scale, wz + 0.05 * scale]
    lms[3] = [wx - 0.55 * scale, wy - 0.75 * scale, wz + 0.08 * scale]
    lms[4] = [wx - 0.60 * scale, wy - 0.95 * scale, wz + 0.10 * scale]

    # Index
    lms[5] = [wx - 0.20 * scale, wy - 0.70 * scale, wz]
    lms[6] = [wx - 0.22 * scale, wy - 1.05 * scale, wz]
    lms[7] = [wx - 0.23 * scale, wy - 1.30 * scale, wz]
    lms[8] = [wx - 0.24 * scale, wy - 1.55 * scale, wz]

    # Middle
    lms[9] = [wx, wy - 0.75 * scale, wz]
    lms[10] = [wx, wy - 1.15 * scale, wz]
    lms[11] = [wx, wy - 1.45 * scale, wz]
    lms[12] = [wx, wy - 1.70 * scale, wz]

    # Ring
    lms[13] = [wx + 0.20 * scale, wy - 0.70 * scale, wz]
    lms[14] = [wx + 0.22 * scale, wy - 1.05 * scale, wz]
    lms[15] = [wx + 0.23 * scale, wy - 1.30 * scale, wz]
    lms[16] = [wx + 0.24 * scale, wy - 1.55 * scale, wz]

    # Pinky
    lms[17] = [wx + 0.38 * scale, wy - 0.60 * scale, wz]
    lms[18] = [wx + 0.42 * scale, wy - 0.88 * scale, wz]
    lms[19] = [wx + 0.45 * scale, wy - 1.10 * scale, wz]
    lms[20] = [wx + 0.48 * scale, wy - 1.30 * scale, wz]

    return lms


def curl_finger(hand: np.ndarray, finger_indices: List[int], curl_factor: float = 0.8):
    """Curls a finger inward towards palm."""
    mcp_idx = finger_indices[0]
    mcp = hand[mcp_idx]
    for idx in finger_indices[1:]:
        # bend towards palm / wrist
        direction = hand[0] - hand[idx]
        hand[idx] += direction * curl_factor * 0.4
        hand[idx, 2] += 0.05 * curl_factor


def get_canonical_sign_pose(sign: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns canonical (left_hand, right_hand) landmarks for a sign.
    Shape: ((21, 3), (21, 3))
    Zeros if a hand is not involved.
    """
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    thumb_idx = [1, 2, 3, 4]
    index_idx = [5, 6, 7, 8]
    middle_idx = [9, 10, 11, 12]
    ring_idx = [13, 14, 15, 16]
    pinky_idx = [17, 18, 19, 20]

    if sign == "NONE":
        # Return zeros or randomly scattered low-activity hands
        return left, right

    # Most single-hand signs use dominant Right hand
    h_r = build_base_hand(wrist=(0.6, 0.6, 0.0))

    if sign == "0":
        # All fingertips curled into 'O' meeting thumb tip
        curl_finger(h_r, thumb_idx, 0.7)
        curl_finger(h_r, index_idx, 0.85)
        curl_finger(h_r, middle_idx, 0.90)
        curl_finger(h_r, ring_idx, 0.90)
        curl_finger(h_r, pinky_idx, 0.90)
        right = h_r

    elif sign == "1":
        # Index extended, others tightly curled
        curl_finger(h_r, thumb_idx, 0.9)
        curl_finger(h_r, middle_idx, 0.95)
        curl_finger(h_r, ring_idx, 0.95)
        curl_finger(h_r, pinky_idx, 0.95)
        right = h_r

    elif sign == "2":
        # Index and Middle extended (V), others curled
        curl_finger(h_r, thumb_idx, 0.9)
        curl_finger(h_r, ring_idx, 0.95)
        curl_finger(h_r, pinky_idx, 0.95)
        # spread index and middle slightly
        h_r[8, 0] -= 0.03
        h_r[12, 0] += 0.03
        right = h_r

    elif sign == "3":
        # Thumb, Index, Middle extended
        curl_finger(h_r, ring_idx, 0.95)
        curl_finger(h_r, pinky_idx, 0.95)
        right = h_r

    elif sign == "4":
        # 4 fingers extended, thumb tucked
        curl_finger(h_r, thumb_idx, 0.95)
        right = h_r

    elif sign == "5":
        # All 5 fingers extended and spread
        h_r[4, 0] -= 0.05
        h_r[8, 0] -= 0.03
        h_r[16, 0] += 0.03
        h_r[20, 0] += 0.05
        right = h_r

    elif sign == "HELLO":
        # Open hand tilted near temple
        h_r[:, 1] -= 0.25
        h_r[:, 0] += 0.10
        right = h_r

    elif sign == "THANK YOU":
        # Flat hand moving from chin forward
        curl_finger(h_r, thumb_idx, 0.3)
        h_r[:, 2] += 0.15
        right = h_r

    elif sign == "YES":
        # Fist nodding downward
        curl_finger(h_r, thumb_idx, 0.9)
        curl_finger(h_r, index_idx, 0.9)
        curl_finger(h_r, middle_idx, 0.9)
        curl_finger(h_r, ring_idx, 0.9)
        curl_finger(h_r, pinky_idx, 0.9)
        h_r[:, 1] += 0.05
        right = h_r

    elif sign == "NO":
        # Index extended wagging or pinch collapse
        curl_finger(h_r, middle_idx, 0.9)
        curl_finger(h_r, ring_idx, 0.9)
        curl_finger(h_r, pinky_idx, 0.9)
        h_r[8, 0] += 0.05
        right = h_r

    elif sign == "HELP":
        # Two hands: Left palm flat horizontal; Right hand thumbs up resting on left palm
        h_l = build_base_hand(wrist=(0.4, 0.65, 0.0))
        # Left palm flat horizontal
        h_l[:, 1] = 0.65
        h_l[:, 2] -= 0.05
        left = h_l

        # Right hand thumbs up fist
        curl_finger(h_r, index_idx, 0.95)
        curl_finger(h_r, middle_idx, 0.95)
        curl_finger(h_r, ring_idx, 0.95)
        curl_finger(h_r, pinky_idx, 0.95)
        h_r[:, 0] = 0.42
        h_r[:, 1] = 0.58
        right = h_r

    elif sign == "WHERE":
        # Open palm facing upward swaying
        h_r[:, 2] -= 0.10
        curl_finger(h_r, thumb_idx, 0.2)
        right = h_r

    elif sign == "WHEN":
        # Index finger pointing up circling
        curl_finger(h_r, thumb_idx, 0.8)
        curl_finger(h_r, middle_idx, 0.9)
        curl_finger(h_r, ring_idx, 0.9)
        curl_finger(h_r, pinky_idx, 0.9)
        h_r[8, 2] += 0.08
        right = h_r

    elif sign == "WATER":
        # 'W' shape (index, middle, ring) near chin
        curl_finger(h_r, thumb_idx, 0.9)
        curl_finger(h_r, pinky_idx, 0.95)
        h_r[:, 1] -= 0.20
        right = h_r

    elif sign == "TOILET":
        # 'T' hand: thumb between index and middle
        curl_finger(h_r, index_idx, 0.7)
        curl_finger(h_r, middle_idx, 0.85)
        curl_finger(h_r, ring_idx, 0.95)
        curl_finger(h_r, pinky_idx, 0.95)
        h_r[4] = h_r[6] + np.array([0.01, -0.01, 0.02])
        right = h_r

    elif sign == "MONEY":
        # Thumb rubbing against index and middle tips
        curl_finger(h_r, ring_idx, 0.95)
        curl_finger(h_r, pinky_idx, 0.95)
        curl_finger(h_r, index_idx, 0.5)
        curl_finger(h_r, middle_idx, 0.5)
        h_r[4] = (h_r[8] + h_r[12]) / 2.0
        right = h_r

    elif sign == "DOCTOR":
        # Left wrist extended, right 3 fingers touching left radial pulse
        h_l = build_base_hand(wrist=(0.35, 0.65, 0.0))
        left = h_l
        # Right index/middle/ring touching left wrist
        curl_finger(h_r, pinky_idx, 0.95)
        curl_finger(h_r, thumb_idx, 0.8)
        h_r[:, 0] = 0.36
        h_r[:, 1] = 0.64
        right = h_r

    elif sign == "TICKET":
        # Index and middle finger clipping / pinching together
        curl_finger(h_r, ring_idx, 0.95)
        curl_finger(h_r, pinky_idx, 0.95)
        curl_finger(h_r, thumb_idx, 0.6)
        h_r[8] = h_r[4] + np.array([0.01, 0.0, 0.01])
        h_r[12] = h_r[4] + np.array([0.02, 0.0, 0.01])
        right = h_r

    elif sign == "TRAIN":
        # Left hand flat (tracks), right index/middle moving forward
        h_l = build_base_hand(wrist=(0.38, 0.65, 0.0))
        left = h_l
        curl_finger(h_r, thumb_idx, 0.9)
        curl_finger(h_r, ring_idx, 0.95)
        curl_finger(h_r, pinky_idx, 0.95)
        h_r[:, 0] = 0.40
        h_r[:, 1] = 0.60
        right = h_r

    elif sign == "PLATFORM":
        # Two hands flat horizontal side-by-side
        h_l = build_base_hand(wrist=(0.35, 0.65, 0.0))
        h_r = build_base_hand(wrist=(0.65, 0.65, 0.0))
        left = h_l
        right = h_r

    elif sign == "PAIN":
        # Two index fingers pointing at each other
        h_l = build_base_hand(wrist=(0.40, 0.60, 0.0))
        curl_finger(h_l, thumb_idx, 0.9)
        curl_finger(h_l, middle_idx, 0.95)
        curl_finger(h_l, ring_idx, 0.95)
        curl_finger(h_l, pinky_idx, 0.95)
        left = h_l

        curl_finger(h_r, thumb_idx, 0.9)
        curl_finger(h_r, middle_idx, 0.95)
        curl_finger(h_r, ring_idx, 0.95)
        curl_finger(h_r, pinky_idx, 0.95)
        h_r[:, 0] = 0.55
        right = h_r

    elif sign == "MEDICINE":
        # Right hand brings pill to mouth
        curl_finger(h_r, middle_idx, 0.9)
        curl_finger(h_r, ring_idx, 0.9)
        curl_finger(h_r, pinky_idx, 0.9)
        h_r[4] = h_r[8] + np.array([0.0, 0.01, 0.01])
        h_r[:, 1] -= 0.25
        right = h_r

    elif sign == "FEVER":
        # Back of right hand touching forehead
        h_r[:, 1] -= 0.35
        h_r[:, 0] = 0.50
        h_r[:, 2] += 0.10
        right = h_r

    elif sign == "HEAD":
        # Right index pointing at temple
        curl_finger(h_r, thumb_idx, 0.9)
        curl_finger(h_r, middle_idx, 0.95)
        curl_finger(h_r, ring_idx, 0.95)
        curl_finger(h_r, pinky_idx, 0.95)
        h_r[:, 1] -= 0.32
        h_r[:, 0] = 0.58
        right = h_r

    elif sign == "STOMACH":
        # Right hand flat against stomach
        h_r[:, 1] += 0.15
        h_r[:, 0] = 0.50
        right = h_r

    return left, right


def rotate_3d(pts: np.ndarray, rx: float, ry: float, rz: float) -> np.ndarray:
    """Applies small 3D rotation in radians."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    center = pts[0]  # rotate around wrist
    return (pts - center) @ R.T + center


def generate_dataset(raw_dir: str = "data/raw", samples_per_sign_per_signer: int = 150):
    """
    Generates synthetic .npz raw dataset files matching the recording format:
    data/raw/<signer>/<sign>.npz
    Each .npz contains:
      'landmarks': shape (samples, 2, 21, 3)
      'sign': string
      'signer': string
    """
    np.random.seed(1337)
    os.makedirs(raw_dir, exist_ok=True)

    print("Generating comprehensive ISL dataset across 26 classes and 3 signers...")

    total_samples = 0
    for signer in SIGNERS:
        signer_dir = os.path.join(raw_dir, signer)
        os.makedirs(signer_dir, exist_ok=True)

        # Signer specific scale and bias
        signer_scale = 1.0 + (SIGNERS.index(signer) - 1) * 0.08

        for sign in CLASSES:
            base_left, base_right = get_canonical_sign_pose(sign)
            samples = []

            for _ in range(samples_per_sign_per_signer):
                # Augmentations: jitter, 3D rotation, scale
                scale_jitter = np.random.uniform(0.90, 1.10) * signer_scale
                rot_x = np.random.uniform(-0.15, 0.15)
                rot_y = np.random.uniform(-0.15, 0.15)
                rot_z = np.random.uniform(-0.15, 0.15)

                cur_left = np.zeros((21, 3), dtype=np.float32)
                cur_right = np.zeros((21, 3), dtype=np.float32)

                if np.any(np.abs(base_left) > 1e-5):
                    l_pts = base_left.copy() * scale_jitter
                    l_pts = rotate_3d(l_pts, rot_x, rot_y, rot_z)
                    l_pts += np.random.normal(0, 0.008, size=l_pts.shape)
                    cur_left = l_pts

                if np.any(np.abs(base_right) > 1e-5):
                    r_pts = base_right.copy() * scale_jitter
                    r_pts = rotate_3d(r_pts, rot_x, rot_y, rot_z)
                    r_pts += np.random.normal(0, 0.008, size=r_pts.shape)
                    cur_right = r_pts

                # Left-handed signer augmentation for 10% of samples
                if np.random.rand() < 0.10:
                    # swap hands and flip x
                    cur_left, cur_right = cur_right.copy(), cur_left.copy()
                    if np.any(np.abs(cur_left) > 1e-5):
                        cur_left[:, 0] = 1.0 - cur_left[:, 0]
                    if np.any(np.abs(cur_right) > 1e-5):
                        cur_right[:, 0] = 1.0 - cur_right[:, 0]

                # Frame shape (2, 21, 3)
                frame = np.stack([cur_left, cur_right], axis=0)
                samples.append(frame)

            samples_arr = np.array(samples, dtype=np.float32)
            npz_path = os.path.join(signer_dir, f"{sign}.npz")
            np.savez_compressed(
                npz_path,
                landmarks=samples_arr,
                sign=sign,
                signer=signer
            )
            total_samples += len(samples)

    print(f"Dataset generation complete! Created {total_samples} samples across {len(CLASSES)} classes in {raw_dir}")


if __name__ == "__main__":
    generate_dataset()
