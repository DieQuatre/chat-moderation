"""Classical baseline (area B): tf-idf features + logistic regression.

Run:  python -m src.models.tfidf_lr      (trains on train, scores val, saves models/tfidf_lr.joblib)
"""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.text.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "models" / "tfidf_lr.joblib"


class TfidfLR:
    name = "tf-idf + LR"

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def train(cls, texts, labels):
        pipeline = make_pipeline(
            TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2,
                            max_features=100_000, sublinear_tf=True),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        pipeline.fit(texts, labels)
        return cls(pipeline)

    def save(self):
        MODEL_FILE.parent.mkdir(exist_ok=True)
        joblib.dump(self.pipeline, MODEL_FILE)

    @classmethod
    def load(cls):
        return cls(joblib.load(MODEL_FILE))

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(list(texts))[:, 1]


if __name__ == "__main__":
    from src.data.load import load_split
    from src.eval.metrics import report

    train_texts, train_labels = load_split("train")
    model = TfidfLR.train(train_texts, train_labels)
    model.save()
    val_texts, val_labels = load_split("val")
    report(model.name, val_labels, model.predict_proba(val_texts))
    print("Saved", MODEL_FILE)
