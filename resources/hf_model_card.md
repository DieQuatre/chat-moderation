---
language: en
tags:
- text-classification
- toxicity
- content-moderation
- onnx
- quantization
pipeline_tag: text-classification
---

# Chat Moderation Models (SWE015 homework)

Trained models for **real-time chat moderation**: each English chat message is classified as
**clean** or **offensive**. Built for the SWE015 *Introduction to Large Language Models* homework
(Use Case 4).

Code, app and full documentation: **https://github.com/DieQuatre/chat-moderation**
The app (`python app.py`) downloads this repository automatically on first run.

## Files

| File | Model |
|---|---|
| `bert-onnx/model_int8.onnx` | Fine-tuned bert-base-uncased, ONNX, dynamic INT8 quantization (used by default in the app) |
| `bert-fp32/` | Fine-tuned bert-base-uncased, FP32 (PyTorch / safetensors) and tokenizer |
| `tfidf_lr.joblib` | tf-idf + logistic regression baseline (scikit-learn) |
| `w2v.model`, `ffn.pt` | word2vec (skip-gram, 100 dims) + feedforward network 100 → 64 → 2 |
| `known_words.txt` | Known-word list used by the lexicon + edit-distance rule |

## Results

Same held-out test set for all models: 10,000 comments from the official labeled Jigsaw test set,
decision threshold 0.5.

| Model | Test macro-F1 | Size (MB) | Latency (ms, mean ± std) |
|---|---|---|---|
| Lexicon rule + edit distance | 0.7294 | 0.2 | 0.15 ± 0.20 |
| tf-idf + logistic regression | 0.7795 | 4.6 | 1.12 ± 0.43 |
| word2vec + feedforward net | 0.7685 | 30.1 | 0.49 ± 0.45 |
| BERT FP32 (PyTorch) | 0.8219 | 417.7 | 113.18 ± 71.89 |
| **BERT INT8 (ONNX Runtime)** | **0.8258** | **105.2** | **21.18 ± 11.93** |

Latency: Intel Core i7-12700H CPU (Windows), batch size 1, 10 warm-up + 100 timed runs.

## Training data

Kaggle *Jigsaw Toxic Comment Classification Challenge* (2018): 40,000 training and 5,000
validation comments from `train.csv`. A comment is labeled offensive if any of the six Jigsaw labels
is 1. Dataset under CC0; comment text under Wikipedia's CC-BY-SA 3.0. The data itself is not
included here.

## Intended use and limitations

For study and demonstration only, not for production moderation.

- **Identity-term bias:** neutral sentences mentioning identity terms can be flagged
  (e.g. 45% of clean test comments containing "gay" were flagged, vs. 7.2% of all clean comments).
- Insults without offensive words can be missed.
- Trained on Wikipedia talk-page comments, not game chat; English only.
