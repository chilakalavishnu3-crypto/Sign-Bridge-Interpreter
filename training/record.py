"""
SignBridge Training Data Recorder
Interactive webcam data collector for capturing authentic ISL hand landmarks.
Uses MediaPipe Tasks Hand Landmarker (hand_landmarker.task) in VIDEO mode.
Matches B1 specification from Prompt Library.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from training.dataset_generator import CLASSES, SIGNERS
except ImportError:
    from dataset_generator import CLASSES, SIGNERS

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


def draw_hand_skeleton(frame, landmarks, color=(0, 240, 255), joint_color=(255, 184, 0)):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pts[start], pts[end], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, joint_color, -1)


def get_dataset_progress(raw_dir: str = "data/raw") -> dict:
    progress = {}
    for signer in SIGNERS:
        progress[signer] = {}
        s_dir = os.path.join(raw_dir, signer)
        for sign in CLASSES:
            npz_path = os.path.join(s_dir, f"{sign}.npz")
            if os.path.exists(npz_path):
                try:
                    data = np.load(npz_path)
                    progress[signer][sign] = len(data["landmarks"])
                except Exception:
                    progress[signer][sign] = 0
            else:
                progress[signer][sign] = 0
    return progress


def print_progress_grid(progress: dict):
    print("\n" + "=" * 65)
    print(f"{'SIGN':<12} | " + " | ".join([f"{s:^12}" for s in SIGNERS]))
    print("-" * 65)
    for sign in CLASSES:
        counts = [f"{progress[s].get(sign, 0):^12}" for s in SIGNERS]
        print(f"{sign:<12} | " + " | ".join(counts))
    print("=" * 65 + "\n")


def record_sign(signer: str, sign: str, target_frames: int = 200, raw_dir: str = "data/raw"):
    signer_dir = os.path.join(raw_dir, signer)
    os.makedirs(signer_dir, exist_ok=True)
    out_file = os.path.join(signer_dir, f"{sign}.npz")

    model_path = os.path.join("assets", "model", "hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    recorded_frames = []
    if os.path.exists(out_file):
        try:
            prev = np.load(out_file)
            recorded_frames = list(prev["landmarks"])
            print(f"Loaded {len(recorded_frames)} existing frames for {sign}.")
        except Exception:
            recorded_frames = []

    state = "READY"
    countdown_start = 0
    clip_frames = 0

    print(f"\nTarget: {sign} for {signer} | Press SPACE to record 30-frame clip. ESC to exit.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_timestamp_ms = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms += 33

            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            left_hand = np.zeros((21, 3), dtype=np.float32)
            right_hand = np.zeros((21, 3), dtype=np.float32)

            if detection_result.hand_landmarks:
                for idx, hand_lms in enumerate(detection_result.hand_landmarks):
                    handedness = detection_result.handedness[idx][0].category_name
                    draw_hand_skeleton(frame, hand_lms,
                                       color=(0, 240, 255) if handedness == "Right" else (255, 0, 180))

                    # Isotropic coords (x, z scaled by width/height) — matches the server pipeline.
                    aspect = frame.shape[1] / frame.shape[0]
                    pts = np.array([[lm.x * aspect, lm.y, lm.z * aspect] for lm in hand_lms], dtype=np.float32)
                    if handedness == "Left":
                        left_hand = pts
                    else:
                        right_hand = pts

            current_time = time.time()

            # UI Header
            cv2.rectangle(frame, (10, 10), (630, 85), (15, 25, 40), -1)
            cv2.putText(frame, f"SignBridge Recorder | Signer: {signer}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2)
            cv2.putText(frame, f"Sign: [{sign}]  Progress: {len(recorded_frames)} / {target_frames}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if state == "READY":
                cv2.putText(frame, "Press SPACE to Record (3s Countdown)", (110, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            elif state == "COUNTDOWN":
                elapsed = current_time - countdown_start
                count = 3 - int(elapsed)
                if count > 0:
                    cv2.putText(frame, str(count), (300, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 200, 255), 4)
                else:
                    state = "RECORDING"
                    clip_frames = 0
            elif state == "RECORDING":
                frame_sample = np.stack([left_hand, right_hand], axis=0)
                recorded_frames.append(frame_sample)
                clip_frames += 1

                cv2.circle(frame, (600, 45), 14, (0, 0, 255), -1)
                cv2.putText(frame, f"RECORDING ({clip_frames}/30)", (200, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                if clip_frames >= 30 or len(recorded_frames) >= target_frames:
                    np.savez_compressed(
                        out_file,
                        landmarks=np.array(recorded_frames, dtype=np.float32),
                        sign=sign,
                        signer=signer
                    )
                    print(f"Saved {len(recorded_frames)} frames to {out_file}")
                    if len(recorded_frames) >= target_frames:
                        print(f"Completed target frames for [{sign}]!")
                        break
                    state = "READY"

            cv2.imshow("SignBridge Data Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32 and state == "READY":
                state = "COUNTDOWN"
                countdown_start = time.time()

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SignBridge ISL Data Recorder")
    parser.add_argument("--grid", action="store_true", help="Print progress grid and exit")
    parser.add_argument("--signer", type=str, default="Signer_A", choices=SIGNERS)
    parser.add_argument("--sign", type=str, default="TICKET", choices=CLASSES)
    args = parser.parse_args()

    progress = get_dataset_progress()
    if args.grid:
        print_progress_grid(progress)
        return

    print_progress_grid(progress)
    record_sign(args.signer, args.sign)


if __name__ == "__main__":
    main()
