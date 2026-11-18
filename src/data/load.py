"""Load the fixed splits made by prepare_jigsaw.py."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_split(name):
    """name: 'train', 'val' or 'test' -> (list of texts, numpy array of labels)."""
    df = pd.read_csv(ROOT / "data" / f"{name}.csv")
    return df["text"].fillna("").tolist(), df["label"].to_numpy()
