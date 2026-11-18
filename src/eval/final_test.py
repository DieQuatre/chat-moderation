"""Paket 5: score all five models on the SAME held-out test set and benchmark them (R3, R4).

For every model:
  - macro-F1 on data/test.csv (10,000 comments, never used for training or tuning)
  - size on disk (MB) of every file the model needs
  - latency per message on CPU: 10 warm-up runs, then 100 timed runs with batch size 1,
    one different test message per run (same method as the lecture PerformanceBenchmark)

Run:  python -m src.eval.final_test
Output: results/test_results.csv, results/test_results.md, data/test_predictions.csv
"""
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import load_split
from src.eval.metrics import macro_f1

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
WARMUP, RUNS = 10, 100
LATENCY_BUDGET_MS = 50


def size_mb(paths):
    total = 0
    for p in paths:
        p = Path(p)
        files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
        total += sum(f.stat().st_size for f in files)
    return total / 2**20


def latency_ms(model, messages):
    for text in messages[:WARMUP]:
        model.predict_proba([text])
    times = []
    for text in messages[:RUNS]:
        start = time.perf_counter()
        model.predict_proba([text])
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.std(times))


def load_models():
    """(model used for test scores, model used for CPU latency, files for size)."""
    import torch

    from src.models.bert import FP32_DIR, ONNX_INT8, BertFP32, BertInt8
    from src.models.lexicon_rule import KNOWN_FILE, LEXICON_FILE, LexiconRule
    from src.models.tfidf_lr import MODEL_FILE, TfidfLR
    from src.models.w2v_ffn import FFN_FILE, W2V_FILE, W2VFFN

    lexicon, tfidf, w2v, int8 = LexiconRule.load(), TfidfLR.load(), W2VFFN.load(), BertInt8()
    bert_cpu = BertFP32(device="cpu")
    bert_score = BertFP32(device="cuda") if torch.cuda.is_available() else bert_cpu  # same weights, faster scoring
    w2v_files = [W2V_FILE] + list(W2V_FILE.parent.glob(W2V_FILE.name + ".*")) + [FFN_FILE]
    return [
        (lexicon, lexicon, [LEXICON_FILE, KNOWN_FILE]),
        (tfidf, tfidf, [MODEL_FILE]),
        (w2v, w2v, w2v_files),
        (bert_score, bert_cpu, [FP32_DIR / "model.safetensors"]),
        (int8, int8, [ONNX_INT8]),
    ]


def main():
    texts, labels = load_split("test")
    rng = np.random.default_rng(42)
    messages = [texts[i] for i in rng.choice(len(texts), WARMUP + RUNS, replace=False)]
    hardware = f"{platform.processor()} ({platform.system()}), CPU only, batch size 1"
    print("Latency hardware:", hardware)

    rows, predictions = [], {"label": labels}
    for scorer, timer, files in load_models():
        print(f"\n{scorer.name}: scoring {len(texts):,} test comments ...", flush=True)
        probs = scorer.predict_proba(texts)
        predictions[scorer.name] = np.round(probs, 4)
        mean, std = latency_ms(timer, messages)
        row = {"model": scorer.name, "test_macro_f1": round(macro_f1(labels, probs), 4),
               "size_mb": round(size_mb(files), 1),
               "latency_ms_mean": round(mean, 2), "latency_ms_std": round(std, 2)}
        print(row, flush=True)
        rows.append(row)

    table = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    table.to_csv(ROOT / "results" / "test_results.csv", index=False)
    md = ["| Model | Test macro-F1 | Size (MB) | Latency (ms, mean ± std) |", "|---|---|---|---|"]
    md += [f"| {r.model} | {r.test_macro_f1:.4f} | {r.size_mb:.1f} | {r.latency_ms_mean:.2f} ± {r.latency_ms_std:.2f} |"
           for r in table.itertuples()]
    md.append(f"\nLatency: {WARMUP} warm-up + {RUNS} timed runs, {hardware}.")
    (ROOT / "results" / "test_results.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(predictions).to_csv(ROOT / "data" / "test_predictions.csv", index=False)

    print("\n" + "\n".join(md))
    int8 = table[table.model == "BERT INT8"].iloc[0]
    verdict = "MET" if int8.latency_ms_mean <= LATENCY_BUDGET_MS else "NOT MET"
    print(f"\nLatency budget {LATENCY_BUDGET_MS} ms for BERT INT8: {verdict} ({int8.latency_ms_mean:.2f} ms)")


if __name__ == "__main__":
    main()
