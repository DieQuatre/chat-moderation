"""Shared scoring: macro-F1 at threshold 0.5, saved to results/val_scores.json."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.5


def macro_f1(labels, probs):
    return f1_score(labels, (np.asarray(probs) >= THRESHOLD).astype(int), average="macro")


def report(model_name, labels, probs, split="val"):
    preds = (np.asarray(probs) >= THRESHOLD).astype(int)
    score = f1_score(labels, preds, average="macro")
    print(f"\n{model_name} on {split}: macro-F1 = {score:.4f}")
    print(classification_report(labels, preds, target_names=["clean", "offensive"], digits=4))
    path = ROOT / "results" / f"{split}_scores.json"
    path.parent.mkdir(exist_ok=True)
    scores = json.loads(path.read_text()) if path.exists() else {}
    scores[model_name] = round(float(score), 4)
    path.write_text(json.dumps(scores, indent=2))
    return score
