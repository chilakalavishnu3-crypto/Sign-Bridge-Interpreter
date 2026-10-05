"""
SignBridge Desktop Native Kiosk (OpenCV + MediaPipe Tasks)
Standalone offline desktop application with direct webcam access,
HUD display, 1-second hold detector, and local audio speech synthesis.
"""

import os
import sys
import time
import json
import joblib
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from backend.session import SignSession
from training.dataset_generator import CLASSES
from training.normalize import correct_aspect, normalize_dual_hands

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_skeleton(frame, landmarks, color=(0, 240, 255)):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, (255, 184, 0), -1)


def main():
    model_path = os.path.join("assets", "model", "isl_model.joblib")
    task_path = os.path.join("assets", "model", "hand_landmarker.task")
    templates_path = os.path.join("assets", "packs", "templates.json")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run training/train.py first.")
        return

    classifier = joblib.load(model_path)
    templates = {}
    if os.path.exists(templates_path):
        with open(templates_path, "r", encoding="utf-8") as f:
            templates = json.load(f)

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Same time-based hold/pause rules as the web kiosk (backend/session.py).
    session = SignSession()
    current_sentence = "..."
    current_lang = "en"  # en, ta, hi

    print("\nStarting SignBridge Desktop Kiosk...")
    print("Controls: [Q] Quit  [R] Reset  [S] Switch Language (en/ta/hi)\n")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_timestamp_ms = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms += 33

            detection = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            hands_data = []
            if detection.hand_landmarks:
                for idx, hand_lms in enumerate(detection.hand_landmarks):
                    label = detection.handedness[idx][0].category_name
                    draw_skeleton(frame, hand_lms,
                                  color=(0, 240, 255) if label == "Right" else (255, 0, 180))
                    hands_data.append({
                        "label": label,
                        "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_lms]
                    })

            # Logic
            pred_sign = "NONE"
            conf = 0.0
            if hands_data:
                norm_feat = normalize_dual_hands(correct_aspect(hands_data, w / h))
                probs = classifier.predict_proba(np.array(norm_feat, dtype=np.float32).reshape(1, -1))[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                pred_sign = CLASSES[pred_idx]

            outcome = session.process(len(hands_data), pred_sign, conf)
            hold_progress = outcome.hold_progress
            confirmed_words = outcome.words
            if outcome.sentence_words:
                sign_key = " ".join(outcome.sentence_words)
                current_sentence = templates.get(sign_key, {}).get(current_lang, sign_key)
                print(f"\n[SENTENCE CONFIRMED]: {current_sentence}")

            # Render Desktop HUD
            # Top Banner
            cv2.rectangle(frame, (0, 0), (w, 80), (10, 20, 32), -1)
            cv2.line(frame, (0, 80), (w, 80), (0, 240, 255), 2)
            cv2.putText(frame, "SignBridge Lite | Public Counter ISL Kiosk", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)

            words_str = " -> ".join(confirmed_words) if confirmed_words else "Waiting for signs..."
            cv2.putText(frame, f"Signs: [{words_str}]", (16, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 184, 0), 2)

            # Hold Meter Bar
            if hold_progress > 0:
                bar_w = int(200 * hold_progress)
                cv2.rectangle(frame, (w - 230, 25), (w - 30, 45), (40, 50, 60), -1)
                cv2.rectangle(frame, (w - 230, 25), (w - 230 + bar_w, 45), (0, 240, 255), -1)
                cv2.putText(frame, f"HOLD {int(hold_progress * 100)}%", (w - 225, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # Bottom Sentence Card
            cv2.rectangle(frame, (0, h - 70), (w, h), (15, 25, 42), -1)
            cv2.line(frame, (0, h - 70), (w, h - 70), (255, 184, 0), 1)
            cv2.putText(frame, f"Lang: [{current_lang.upper()}]  Request:", (16, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1)
            cv2.putText(frame, current_sentence, (16, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("SignBridge Lite Desktop", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                session.reset()
                current_sentence = "..."
            elif key == ord('s'):
                langs = ["en", "ta", "hi"]
                current_lang = langs[(langs.index(current_lang) + 1) % len(langs)]
                print(f"Switched language to: {current_lang}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
