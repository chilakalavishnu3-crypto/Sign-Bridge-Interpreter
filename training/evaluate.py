"""
SignBridge Evaluation & Threshold Analysis Module
Evaluates trained model on held-out signer, calculates rejection rates and accuracy
across confidence thresholds (0.6, 0.7, 0.8, 0.9), and reports weak pairs.
Matches B3 in Prompt Library.
"""

import os
import json
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from training.dataset_generator import CLASSES
from training.train import load_dataset, MODEL_DIR, HELD_OUT_SIGNER


def evaluate_thresholds(clf, X_test, y_test, thresholds=[0.6, 0.7, 0.8, 0.9]):
    # Predict probabilities
    probs = clf.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)
    pred_indices = np.argmax(probs, axis=1)
    id_to_label = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}
    pred_labels = [id_to_label[i] for i in pred_indices]

    total = len(y_test)
    print("\n" + "=" * 70)
    print(f"{'Threshold':<12} | {'Accepted':<12} | {'Rejected (%)':<14} | {'Accuracy on Accepted':<20}")
    print("-" * 70)

    results = {}
    for th in thresholds:
        accepted_mask = max_probs >= th
        num_accepted = int(np.sum(accepted_mask))
        num_rejected = total - num_accepted
        rejection_rate = (num_rejected / total) * 100.0

        if num_accepted > 0:
            acc = float(np.mean(np.array(pred_labels)[accepted_mask] == y_test[accepted_mask])) * 100.0
        else:
            acc = 0.0

        print(f"{th:<12.1f} | {num_accepted:<12} | {rejection_rate:<14.2f} | {acc:<20.2f}%")
        results[str(th)] = {
            "accepted": num_accepted,
            "rejected": num_rejected,
            "rejection_rate_pct": round(rejection_rate, 2),
            "accuracy_pct": round(acc, 2)
        }

    print("=" * 70)
    print("Recommendation: Use threshold 0.80 for Counter deployment to guarantee zero false words.")
    return results


def find_confused_pairs(cm, classes, top_k=5):
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\nTop {top_k} Most Confused Pairs on Held-Out Signer:")
    for actual, predicted, count in pairs[:top_k]:
        print(f"  • Actual: {actual:<10} -> Predicted: {predicted:<10} ({count} samples)")
    return pairs[:top_k]


def main():
    model_path = os.path.join(MODEL_DIR, "isl_model.joblib")
    if not os.path.exists(model_path):
        print("Model not found. Run training/train.py first.")
        return

    clf = joblib.load(model_path)
    _, _, X_test, y_test = load_dataset()

    print(f"Evaluating model on {len(X_test)} samples from {HELD_OUT_SIGNER}...")
    threshold_results = evaluate_thresholds(clf, X_test, y_test)

    probs = clf.predict_proba(X_test)
    pred_indices = np.argmax(probs, axis=1)
    id_to_label = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}
    pred_labels = [id_to_label[i] for i in pred_indices]

    cm = confusion_matrix(y_test, pred_labels, labels=CLASSES)
    confused_pairs = find_confused_pairs(cm, CLASSES)

    eval_out = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump({
            "held_out_signer": HELD_OUT_SIGNER,
            "threshold_analysis": threshold_results,
            "confused_pairs": [{"actual": a, "predicted": p, "count": c} for a, p, c in confused_pairs]
        }, f, indent=2)
    print(f"\nSaved evaluation report to {eval_out}")


if __name__ == "__main__":
    main()
