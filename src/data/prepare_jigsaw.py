"""Paket 1: build the fixed train / val / test splits from the Jigsaw data.

Label rule: offensive (1) if ANY of the six Jigsaw labels is 1, otherwise clean (0).
  train: 40,000 comments from train.csv      (stratified, seed 42)
  val:    5,000 other comments from train.csv
  test:  10,000 comments from the official labeled test set (shared by all models)

Run from the project folder:   python -m src.data.prepare_jigsaw
Output: data/train.csv, data/val.csv, data/test.csv  (columns: id, text, label)
        results/data_stats.json
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42
N_TRAIN, N_VAL, N_TEST = 40_000, 5_000, 10_000


def jigsaw_dir():
    """Folder with the Kaggle files (downloaded once, then read from the kagglehub cache)."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    token_file = Path.home() / ".kaggle" / "access_token"
    if "KAGGLE_API_TOKEN" not in os.environ and token_file.exists():
        os.environ["KAGGLE_API_TOKEN"] = token_file.read_text().strip()
    import kagglehub

    return Path(kagglehub.competition_download("jigsaw-toxic-comment-classification-challenge"))


def read(folder, name):
    """Read name.csv, or name.csv.zip if it is still zipped."""
    path = folder / name
    return pd.read_csv(path if path.exists() else folder / f"{name}.zip")


def to_binary(df):
    return pd.DataFrame({
        "id": df["id"],
        "text": df["comment_text"],
        "label": (df[LABELS].max(axis=1) > 0).astype(int),
    })


def sample(df, n):
    part, rest = train_test_split(df, train_size=n, stratify=df["label"], random_state=SEED)
    return part.reset_index(drop=True), rest


def stats(df):
    n_off = int(df["label"].sum())
    return {"comments": len(df), "offensive": n_off, "clean": len(df) - n_off,
            "offensive_ratio": round(n_off / len(df), 4)}


def main():
    src = jigsaw_dir()
    train_full = to_binary(read(src, "train.csv"))

    test_full = read(src, "test.csv").merge(read(src, "test_labels.csv"), on="id")
    test_full = test_full[test_full["toxic"] != -1]  # -1 = not scored by Kaggle, no real label
    test_full = to_binary(test_full)

    train, rest = sample(train_full, N_TRAIN)
    val, _ = sample(rest, N_VAL)
    test, _ = sample(test_full, N_TEST)

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(data_dir / f"{name}.csv", index=False)

    result = {
        "source": "Kaggle Jigsaw Toxic Comment Classification Challenge (2018)",
        "label_rule": "offensive = any of " + ", ".join(LABELS),
        "seed": SEED,
        "available": {"train.csv": stats(train_full), "labeled test": stats(test_full)},
        "splits": {"train": stats(train), "val": stats(val), "test": stats(test)},
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_stats.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("Saved data/train.csv, data/val.csv, data/test.csv")


if __name__ == "__main__":
    main()
